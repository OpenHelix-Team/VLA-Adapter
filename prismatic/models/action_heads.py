"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS



def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations



class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
        use_3d=False, 
        dim_3d=None,
        inject_layers=None
    ):
        super().__init__()
        self.use_3d = use_3d
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        if not self.use_3d:
            self.model = MLPResNet(
                num_blocks=24, 
                input_dim=input_dim*ACTION_DIM, 
                hidden_dim=hidden_dim, 
                output_dim=action_dim,
                use_pro_version=use_pro_version
                )
        else:
            assert dim_3d is not None, "dim_3d must be specified when use_3d is True!"
            self.model = MLPResNetw3d(
                num_blocks=24, 
                input_dim=input_dim*ACTION_DIM, 
                hidden_dim=hidden_dim, 
                output_dim=action_dim,
                use_pro_version=use_pro_version,
                feat_3d_dim=dim_3d,
                inject_layers=inject_layers
                )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference",
            **kwargs
            ):
        """
        * action_hidden_states: [B, Hidden, L_v + L_a, Dim]
        * proprio_hidden_states: 
        * proprio_projector: [B, P_dim] --> [B, 1, Dim]
        * 输出时：task_hidden_states: [B, Hidden, L_v, Dim], action_hidden_states: [B, Hidden, L_a, Dim]
        * cond_actions_hidden_states: [B, A_dim * A_chunk, Dim] --(reshape)-- [B, A_chunk, A_dim * Dim]
        * 这 rearranged_actions_hidden_states 是 Learnable PE
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)
            print("-----------------")
        if not self.use_3d:
            action = self.model(
                rearranged_actions_hidden_states,
                h_a=actions_hidden_states,
                p=proprio_features,
                h_t=task_hidden_states
                )
        else:
            h_3d = kwargs.get("hidden_3d", None)
            assert h_3d is not None, "h_3d must be passed when use_3d is True!"
            action = self.model(
                rearranged_actions_hidden_states,
                h_a=actions_hidden_states,
                p=proprio_features,
                h_t=task_hidden_states,
                h_3d=h_3d
                )

        return action
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p= None):
        #* [B, A_chunk, A_dim * Dim] -> [B, A_chunk, Dim] -> [B, A_chunk, A_dim]
        #* 每一个 block 内部的过程是：
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x   

class MLPResNetw3d(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=True,
            feat_3d_dim=2048,
            inject_layers=0
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        # if use_3d_feat:
        self.feat_3d_dim = feat_3d_dim
        self.feat_3d_align = nn.Sequential(
            nn.LayerNorm(self.feat_3d_dim),
            nn.Linear(self.feat_3d_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.inject_layers = inject_layers # TODO: inject 3D feat in only one layer
        for i in range(num_blocks):
            if self.inject_layers == "all":
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro_w3d(dim=hidden_dim))
            elif isinstance(self.inject_layers, int) and i == self.inject_layers:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro_w3d(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p= None, h_3d=None):
        #* [B, A_chunk, A_dim * Dim] -> [B, A_chunk, Dim] -> [B, A_chunk, A_dim]
        #* 每一个 block 内部的过程是：
        # x: (batch_size, input_dim)
        h_3d = self.feat_3d_align(h_3d)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            if isinstance(block, MLPResNetBlock_Pro_w3d):
                x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p, h_3d=h_3d)  # shape: (batch_size, hidden_dim)
            elif isinstance(block, MLPResNetBlock_Pro):
                x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        * x: [B, A_chunk, Dim]
        * h_a: [B, L_a, Dim]
        * h_t: [B, L_v, Dim]
        * p:   [B, 1, Dim]
        * 三种：[B, n, A_chunk, dim], [B, n, L_a + p, dim], [B, n, L_v, dim] MHA 方式，加入 RoPE
        * [B, n, A_chunk, dim] 的 q 和 自身的 k、h_t 的 k、h_a 的 k 分别做点积，得到三个
        * [B, n, A_chunk, A_chunk], [B, n, A_chunk, L_a + p], [B, n, A_chunk, L_v] , cat 就是 [B, n, A_chunk, A_chunk + (L_a + p) + L_v]
        * 而 v 三者 cat 在一起就是 [B, n, A_chunk + (L_a + p) + L_v, dim] --> [B, n, A_chunk, dim]
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1)

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens,v_adapter,v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)
        return x

class MLPResNetBlock_Pro_w3d(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim: int, num_heads: int=8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.k_3d = nn.Linear(dim, dim)
        self.v_3d = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None, h_3d=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        * x: [B, A_chunk, Dim]
        * h_a: [B, L_a, Dim]
        * h_t: [B, L_v, Dim]
        * p:   [B, 1, Dim]
        * 三种：[B, n, A_chunk, dim], [B, n, L_a + p, dim], [B, n, L_v, dim] MHA 方式，加入 RoPE
        * [B, n, A_chunk, dim] 的 q 和 自身的 k、h_t 的 k、h_a 的 k 分别做点积，得到三个
        * [B, n, A_chunk, A_chunk], [B, n, A_chunk, L_a + p], [B, n, A_chunk, L_v] , cat 就是 [B, n, A_chunk, A_chunk + (L_a + p) + L_v]
        * 而 v 三者 cat 在一起就是 [B, n, A_chunk + (L_a + p) + L_v, dim] --> [B, n, A_chunk, dim]
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1)
        

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0
        K_3d = h_3d.size(1) if h_3d is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)
        
        # 3D tokens
        k_3d = self.k_3d(h_3d)
        v_3d = self.v_3d(h_3d)


        # reshape -> multi-head
        def reshape_heads(t: torch.Tensor, B: int, L: int) -> torch.Tensor:
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)
        k_3d, v_3d = reshape_heads(k_3d, B, K_3d), reshape_heads(v_3d, B, K_3d)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)
        cos3d, sin3d = self.rope(seq_len=K_3d, device=x.device, dtype=x.dtype)
        _, k_3d = apply_rope(k_3d, k_3d, cos3d, sin3d)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores.append(torch.matmul(q_1, k_3d.transpose(-2, -1)))
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens, v_adapter, v_task, v_3d]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)
        return x