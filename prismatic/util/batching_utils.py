"""
batching_utils.py

Core definitions of (Distributed) Samplers for VLM finetuning; provides functionality for construction and allocating
"split-modality" batches as described in the LLaVa paper; this makes sure that a given device/batch is either entirely
(vision, language) or (language-only) data, which leads to sizeable efficiency gains.
"""

import math
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


# High-Fidelity Bitwise Reproduction of the LLaVa Codebase Sampler Strategy + Per-Rank Allocation Scheme (following
#   the default batching behavior of HF's Trainer Class --> derived from `accelerate`).
#
#   =>> Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L60
#   =>> Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L603
class SplitModalitySampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        modality_lengths: List[Tuple[bool, int]],
        global_batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        self.seed, self.epoch = seed, 0

        # Custom Parameters
        self.dataset, self.modality_lengths, self.drop_last = dataset, modality_lengths, drop_last
        self.global_batch_size = global_batch_size

        # For our purposes, `drop_last` is always False!
        assert not self.drop_last, "SplitModalitySampler must set `drop_last = False`!"
        if len(self.dataset) == 0:
            raise ValueError("SplitModalitySampler requires a non-empty dataset")
        if len(self.modality_lengths) != len(self.dataset):
            raise ValueError("modality_lengths must contain one entry per dataset item")
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.global_batch_size % self.num_replicas != 0:
            raise ValueError("global_batch_size must be divisible by num_replicas")

        # Each modality is batched independently so that every global batch is
        # homogeneous.  Consequently, two modality tails may require more
        # padding than ceil(len(dataset) / global_batch_size) would provide.
        modality_counts = [
            sum(1 for is_multimodal, _ in self.modality_lengths if is_multimodal),
            sum(1 for is_multimodal, _ in self.modality_lengths if not is_multimodal),
        ]
        num_batches = sum(
            math.ceil(count / self.global_batch_size) for count in modality_counts if count > 0
        )
        self.total_size = num_batches * self.global_batch_size
        self.num_samples = self.total_size // self.num_replicas

    @staticmethod
    def reindex_batch(batch_idxs: List[int], idx2lengths: List[int], n_buckets: int) -> List[List[int]]:
        """Re-indexes a batch in a way that is conducive to DistributedSampler + grouping by seqlen per rank."""
        assert len(batch_idxs) % n_buckets == 0, "Batch length is not divisible by `num_replicas`!"

        # Establish initial buckets, capacities, and max number of elements per bucket
        n_examples_per_bucket = len(batch_idxs) // n_buckets
        bucket_indices = [[] for _ in range(n_buckets)]
        bucket_lengths = [0 for _ in range(n_buckets)]

        # Note that `batch_idxs` is already sorted by corresponding length (in descending order)
        for idx in batch_idxs:
            shortest_bucket_idx = bucket_lengths.index(min(bucket_lengths))
            bucket_indices[shortest_bucket_idx].append(idx)

            # Update `bucket_lengths` --> set length to infinity if at capacity!
            bucket_lengths[shortest_bucket_idx] += idx2lengths[idx]
            if len(bucket_indices[shortest_bucket_idx]) == n_examples_per_bucket:
                bucket_lengths[shortest_bucket_idx] = float("inf")

        return bucket_indices

    def get_modality_and_length_grouped_indices(self, generator: torch.Generator) -> List[int]:
        """
        Returns a list of indices so that each slice of `global_batch_size` consecutive indices corresponds to elements
        of the same modality with each sub-sequence of `per_replica_batch_size` (the batch size each unique device sees
        during distributed training) is roughly grouped by sequence length (for training efficiency).
        """
        multimodal_split = [
            (idx, length) for idx, (is_multimodal, length) in enumerate(self.modality_lengths) if is_multimodal
        ]
        unimodal_split = [
            (idx, length) for idx, (is_multimodal, length) in enumerate(self.modality_lengths) if not is_multimodal
        ]
        if multimodal_split:
            multimodal_indices, multimodal_lengths = map(list, zip(*multimodal_split))
        else:
            multimodal_indices, multimodal_lengths = [], []
        if unimodal_split:
            unimodal_indices, unimodal_lengths = map(list, zip(*unimodal_split))
        else:
            unimodal_indices, unimodal_lengths = [], []

        # We're going to be running sorting/grouping relative to `self.global_batch_size` and `self.num_replicas`
        g_bsz = self.global_batch_size

        def make_batches(indices: List[int], lengths: List[int]) -> List[List[int]]:
            """Shuffle, pad, and distribute one modality without empty-tail failures."""
            if not indices:
                return []

            shuffled_idxs = torch.randperm(len(indices), generator=generator).tolist()
            batch_idxs = [shuffled_idxs[i : i + g_bsz] for i in range(0, len(shuffled_idxs), g_bsz)]

            # Pad a short tail cyclically.  A modality can contain fewer than
            # one global batch, so slicing the first batch once is insufficient.
            if len(batch_idxs[-1]) < g_bsz:
                n_missing = g_bsz - len(batch_idxs[-1])
                first_batch = batch_idxs[0]
                batch_idxs[-1].extend(first_batch[i % len(first_batch)] for i in range(n_missing))

            sorted_batch_idxs = [sorted(batch, key=lambda i: lengths[i], reverse=True) for batch in batch_idxs]
            length_bucketed_idxs = [
                self.reindex_batch(batch, lengths, self.num_replicas) for batch in sorted_batch_idxs
            ]
            output_idxs = [idx for batch in length_bucketed_idxs for bucket in batch for idx in bucket]
            reindexed = [indices[idx] for idx in output_idxs]
            return [reindexed[i : i + g_bsz] for i in range(0, len(reindexed), g_bsz)]

        # IMPORTANT :: At this point, for each modality, we have a list of "batches" (made up of indices) where indices
        # are sorted by example sequence length *within* each batch. To make this more concrete, consider the following:
        #   => World Size (`num_replicas`) = 2
        #   => Global Batch Size (`g_bsz`) = 4
        #   => `multimodal_indices` = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11]
        #      `multimodal_lengths` = [20, 90, 21, 22, 91, 18, 89, 19, 93, 88, 92, 17]
        #
        # At this point in the code, `mm_sorted_batch_idxs` might then look like the following (length in parenthesis):
        #   => `mm_sorted_batch_idxs`: [
        #       [4  (91), 3  (21), 0  (20), 5  (18)]    => Batch 1
        #       [6  (89), 9  (88), 7  (19), 11 (17)]    => Batch 2
        #       [8  (93), 10 (92), 1  (90), 2  (21)]    => Batch 3
        #   ]
        #
        # In practice: `g_bsz` is large (= 128), and for contiguous mini-batch "slices", length variance is low.

        # PROBLEM :: We want to split these "global batches" into equal-sized pieces, so that each "replica" (GPU)
        # sees a "mini-batch" of roughly the same sequence lengths; this is super useful for efficient training.

        # HOWEVER :: The default "access pattern" for splitting a large batch into mini-batches by a DistributedSampler
        # is akin to a "take every k" where `k` is equal to the number of replicas (GPUs) you're training on. Or, in
        # Python notation --> `rank_k_indices = flatten(mm_sorted_batch_idxs)[k::num_replicas].
        #
        # Naively translating this our example means each GPU (in our world of 2 total) sees the following indices
        # (grouped by "mini-batch" = `g_bsz / num_replicas` = 2 for convenience):
        #   => `rank_0_indices`: [ [4 (91), 0 (20)] =>> [6 (89), 7  (19)] =>> [8  (93), 1 (90)] ]
        #   => `rank_1_indices`: [ [3 (21), 5 (18)] =>> [9 (88), 11 (17)] =>> [10 (92), 2 (21)] ]
        #
        # We get lucky sometimes, but for the most part, each "mini-batch" has VASTLY DIFFERENT lengths! Bad!

        # FIX :: If we "undo" the access pattern with the following code and re-arrange the way we allocate batches
        # inside the __iter__ method below, we can allocate indices appropriately. Running the following code gives us
        # the following indices (grouped by "mini-batch" again for convenience):
        #   => `rank_0_indices`: [ [4 (91), 3 (21)] =>> [6  (89), 9 (88)] =>> [8 (93), 10 (92)] ]
        #   => `rank_1_indices`: [ [5 (18), 0 (20)] =>> [11 (17), 7 (19)] =>> [2 (21),  1 (90)] ]
        #
        # Much better! As `g_bsz` and `dataset` grow, we're more often than not getting *decent* groupings!
        mm_batches = make_batches(multimodal_indices, multimodal_lengths)
        uni_batches = make_batches(unimodal_indices, unimodal_lengths)

        # Finally, randomly permute the multimodal & unimodal batches, merging into a single stream of indices
        merged_batches = mm_batches + uni_batches
        merge_idxs = torch.randperm(len(merged_batches), generator=generator)
        all_batches = [merged_batches[idx] for idx in merge_idxs]

        # [Quality of Life] Shift "max length" batch to index 0 --> if we OOM, it happens immediately!
        all_lengths = [length + ((_n_patches := 24 * 24) if is_mm else 0) for is_mm, length in self.modality_lengths]
        all_batches_max_lengths = []
        for batch in all_batches:
            all_batches_max_lengths.append(max([all_lengths[idx] for idx in batch]))

        # Identify Batch with "max length" --> Swap into Index 0
        longest_batch_idx = np.argmax(all_batches_max_lengths)
        all_batches[0], all_batches[longest_batch_idx] = all_batches[longest_batch_idx], all_batches[0]

        # Flatten & Return all Indices
        indices = [idx for batch in all_batches for idx in batch]
        return indices

    def __iter__(self) -> Iterator:
        """Deterministically shuffle, then split indices by modality and length."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.get_modality_and_length_grouped_indices(g)
        assert len(set(indices)) == len(self.modality_lengths) == len(self.dataset), "Oops!"
        assert (len(indices) % self.global_batch_size == 0) and (len(indices) % self.num_replicas) == 0, "Oops"

        # Note :: We compute per-replica batch size as a function of `global_batch` and `num_replicas` to ensure that
        # gradient accumulation doesn't affect what indices are assigned a given rank.
        per_replica_batch_size = self.global_batch_size // self.num_replicas

        # Tensorize & Unravel --> rather than yielding via a `take_every` --> we want to partition a global batch
        # across replicas by assigning each a contiguous sub-sequence.
        indices_t = torch.as_tensor(indices)
        per_replica_batch_indices_t = indices_t.reshape(-1, per_replica_batch_size)
        replica_indices_t = per_replica_batch_indices_t[self.rank :: self.num_replicas]

        replica_indices = replica_indices_t.flatten().tolist()
        return iter(replica_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """To be called *between* epochs, prior to DataLoader instantiation; ensures random order across epochs."""
        self.epoch = epoch
