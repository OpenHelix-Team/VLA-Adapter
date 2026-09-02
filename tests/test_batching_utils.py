"""Regression tests for distributed split-modality batching."""

import importlib.util
from pathlib import Path

import pytest
import torch


_MODULE_PATH = Path(__file__).parents[1] / "prismatic" / "util" / "batching_utils.py"
_SPEC = importlib.util.spec_from_file_location("vla_adapter_batching_utils", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
SplitModalitySampler = _MODULE.SplitModalitySampler


class _Dataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def _sampler(modality_lengths, *, rank=0, global_batch_size=4, num_replicas=2):
    return SplitModalitySampler(
        _Dataset(len(modality_lengths)),
        modality_lengths,
        global_batch_size=global_batch_size,
        num_replicas=num_replicas,
        rank=rank,
        seed=17,
    )


def _global_indices(sampler):
    generator = torch.Generator().manual_seed(sampler.seed + sampler.epoch)
    return sampler.get_modality_and_length_grouped_indices(generator)


def test_all_multimodal_split_handles_short_tail_and_distributes_every_item():
    modality_lengths = [(True, 7), (True, 3), (True, 5)]
    rank0 = _sampler(modality_lengths, rank=0)
    rank1 = _sampler(modality_lengths, rank=1)

    assert len(rank0) == len(rank1) == 2
    assert set(iter(rank0)) | set(iter(rank1)) == {0, 1, 2}
    assert len(rank0) == rank0.num_samples == 2
    assert rank0.total_size == 4


def test_all_unimodal_split_no_longer_crashes_on_empty_multimodal_partition():
    modality_lengths = [(False, 8), (False, 2), (False, 6)]
    sampler = _sampler(modality_lengths)

    indices = _global_indices(sampler)
    assert len(indices) == sampler.total_size == 4
    assert set(indices) == {0, 1, 2}
    assert len(list(iter(sampler))) == sampler.num_samples == 2


def test_mixed_modalities_get_independent_full_batches_and_consistent_length():
    modality_lengths = [(True, 9), (False, 4), (False, 2)]
    sampler = _sampler(modality_lengths)
    indices = _global_indices(sampler)

    assert len(indices) == sampler.total_size == 8
    for batch_start in range(0, len(indices), sampler.global_batch_size):
        batch = indices[batch_start : batch_start + sampler.global_batch_size]
        assert len({modality_lengths[idx][0] for idx in batch}) == 1
    assert set(indices) == {0, 1, 2}
    assert len(list(iter(sampler))) == sampler.num_samples == 4


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"global_batch_size": 0}, "global_batch_size"),
        ({"global_batch_size": 3, "num_replicas": 2}, "divisible"),
    ],
)
def test_invalid_batch_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _sampler([(True, 1)], **kwargs)


def test_metadata_must_match_dataset_length():
    with pytest.raises(ValueError, match="one entry"):
        SplitModalitySampler(_Dataset(2), [(True, 1)], global_batch_size=2, num_replicas=1, rank=0)


def test_empty_dataset_is_rejected_before_sampling():
    with pytest.raises(ValueError, match="non-empty"):
        SplitModalitySampler(_Dataset(0), [], global_batch_size=2, num_replicas=1, rank=0)
