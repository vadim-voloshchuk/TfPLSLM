"""Mamba-2-style recurrent LM with a causal, chunk-updated slot memory.

The SSD equations and initialization follow Mamba-2 (Dao & Gu, 2024).
Training uses the unmodified, pinned upstream Triton SSD scan. The slow reference
scan is for correctness tests only. Explicit memory is read before a chunk and
written after it, so its summary cannot leak future tokens into that chunk.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .kernels import load_ssd


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 512
    layers: int = 16
    d_state: int = 64
    expand: int = 2
    head_dim: int = 64
    ff_dim: int = 1536
    conv_width: int = 4
    scan_chunk: int = 128
    memory_slots: int = 16
    memory_dim: int = 256
    memory_chunk: int = 512
    backend: str = "triton"
    architecture: str = "ssm"


def tree_map(fn, value):
    if isinstance(value, torch.Tensor): return fn(value)
    if isinstance(value, list): return [tree_map(fn, x) for x in value]
    if isinstance(value, tuple): return tuple(tree_map(fn, x) for x in value)
    if isinstance(value, dict): return {k: tree_map(fn, v) for k, v in value.items()}
    return value


def detach_state(state): return tree_map(lambda x: x.detach(), state)


def reference_scan(x, dt, A, B, C, D, initial):
    """Token loop allowed only for unit tests and tiny CPU diagnostics."""
    state = initial.float()
    outputs = []
    for t in range(x.shape[1]):
        decay = torch.exp(dt[:, t].float() * A.float())
        state = state * decay[:, :, None, None] + (
            x[:, t].float()[:, :, :, None] * B[:, t, 0].float()[:, None, None, :]
            * dt[:, t].float()[:, :, None, None])
        y = (state * C[:, t, 0].float()[:, None, None, :]).sum(-1)
        outputs.append(y + x[:, t].float() * D.float()[None, :, None])
    return torch.stack(outputs, 1).to(x.dtype), state


class StateBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, inner = cfg.d_model, cfg.d_model * cfg.expand
        self.inner, self.heads = inner, inner // cfg.head_dim
        convdim = inner + 2 * cfg.d_state
        self.norm = nn.RMSNorm(d)
        self.proj = nn.Linear(d, inner + convdim + self.heads, bias=False)
        self.conv = nn.Conv1d(convdim, convdim, cfg.conv_width, groups=convdim)
        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(.001), math.log(.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.heads))
        self.out_norm = nn.RMSNorm(inner)
        self.out = nn.Linear(inner, d, bias=False)
        self.ff_norm = nn.RMSNorm(d)
        self.ff_in = nn.Linear(d, 2 * cfg.ff_dim, bias=False)
        self.ff_out = nn.Linear(cfg.ff_dim, d, bias=False)
        self.scan = load_ssd() if cfg.backend == "triton" else None

    def initial(self, batch, device, dtype):
        return (torch.zeros(batch, self.heads, self.cfg.head_dim, self.cfg.d_state,
                            device=device, dtype=torch.float32),
                torch.zeros(batch, self.inner + 2 * self.cfg.d_state,
                            self.cfg.conv_width - 1, device=device, dtype=dtype))

    def forward(self, x, state):
        cfg = self.cfg
        ssm, history = state
        z, raw, dt = self.proj(self.norm(x)).split(
            [self.inner, self.inner + 2 * cfg.d_state, self.heads], dim=-1)
        raw = torch.cat((history.to(raw.dtype), raw.transpose(1, 2)), dim=-1)
        history = raw[:, :, -(cfg.conv_width - 1):].contiguous()
        convolved = F.silu(self.conv(raw).transpose(1, 2))
        u, b, c = convolved.split([self.inner, cfg.d_state, cfg.d_state], dim=-1)
        u = u.reshape(*u.shape[:2], self.heads, cfg.head_dim)
        b, c = b.unsqueeze(2), c.unsqueeze(2)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        a = -self.A_log.float().exp()
        if self.scan is None:
            y, new_ssm = reference_scan(u, dt, a, b, c, self.D, ssm)
        else:
            y, new_ssm = self.scan(u, dt, a, b, c, chunk_size=cfg.scan_chunk,
                                  D=self.D, initial_states=ssm.contiguous(),
                                  return_final_states=True)
        y = y.flatten(2)
        x = x + self.out(self.out_norm(y * F.silu(z)))
        gate, val = self.ff_in(self.ff_norm(x)).chunk(2, dim=-1)
        x = x + self.ff_out(F.silu(gate) * val)
        return x, (new_ssm, history)


class LatentMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.slots, self.dim = cfg.memory_slots, cfg.memory_dim
        self.read = nn.Linear(self.slots * self.dim, cfg.d_model, bias=False)
        self.read_gate = nn.Linear(cfg.d_model, cfg.d_model)
        self.write = nn.Linear(cfg.d_model, self.slots * self.dim)
        self.gates = nn.Linear(cfg.d_model, 2 * self.slots)
        self.summary_norm = nn.RMSNorm(cfg.d_model)
        nn.init.constant_(self.gates.bias[:self.slots], 2.)
        nn.init.constant_(self.gates.bias[self.slots:], 0.)

    def inject(self, x, memory):
        read = self.read(memory.flatten(1).to(x.dtype))
        return x + torch.sigmoid(self.read_gate(x)) * read[:, None, :]

    def update(self, summary, memory):
        summary = self.summary_norm(summary)
        retain, write = self.gates(summary).float().sigmoid().chunk(2, -1)
        value = self.write(summary).float().tanh().reshape(-1, self.slots, self.dim)
        updated = retain[..., None] * memory + (1-retain[..., None]) * write[..., None] * value
        return updated, {"retain": retain, "write": write}


class PersistentLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(StateBlock(cfg) for _ in range(cfg.layers))
        self.memory = LatentMemory(cfg) if cfg.memory_slots else None
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=.02)
        for block in self.blocks:
            nn.init.normal_(block.out.weight, std=.02/math.sqrt(2*cfg.layers))
            nn.init.normal_(block.ff_out.weight, std=.02/math.sqrt(2*cfg.layers))

    def initial_state(self, batch, device=None, dtype=None):
        device = device or self.embedding.weight.device
        dtype = dtype or self.embedding.weight.dtype
        return {"layers": [b.initial(batch, device, dtype) for b in self.blocks],
                "memory": torch.zeros(batch, self.cfg.memory_slots, self.cfg.memory_dim,
                                      device=device, dtype=torch.float32),
                "summary": torch.zeros(batch, self.cfg.d_model, device=device), "position": 0}

    def forward(self, ids, state=None, disable_memory=False, return_hidden=False):
        state = state if state is not None else self.initial_state(ids.shape[0])
        remaining = self.cfg.memory_chunk - state["position"]
        if ids.shape[1] > remaining:
            raise ValueError("Split input at fixed memory-chunk boundaries")
        x = self.embedding(ids)
        memory = state["memory"]
        if self.memory is not None and not disable_memory:
            x = self.memory.inject(x, memory)
        layer_states = []
        for block, old in zip(self.blocks, state["layers"]):
            x, new = block(x, old)
            layer_states.append(new)
        hidden = self.norm(x)
        summary = state["summary"] + hidden.float().sum(1)
        pos = state["position"] + ids.shape[1]
        diagnostics = {}
        if pos == self.cfg.memory_chunk:
            if self.memory is not None and not disable_memory:
                memory, diagnostics = self.memory.update(summary / pos, memory)
            summary = torch.zeros_like(summary)
            pos = 0
        new_state = {"layers": layer_states, "memory": memory, "summary": summary, "position": pos}
        return (hidden if return_hidden else self.lm_head(hidden)), new_state, diagnostics

    def loss(self, hidden, targets):
        return F.cross_entropy(self.lm_head(hidden).float().flatten(0, 1), targets.flatten(),
                               ignore_index=-100)


class TransformerLM(nn.Module):
    """Parameter-matched local-context baseline. No cross-chunk history."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.memory_chunk, cfg.d_model)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            cfg.d_model, 8, dim_feedforward=cfg.ff_dim * 2, dropout=0.,
            activation="gelu", batch_first=True, norm_first=True) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=.02)

    def initial_state(self, *args, **kwargs): return None

    def forward(self, ids, state=None, disable_memory=False, return_hidden=False):
        x = self.embedding(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))[None]
        mask = torch.ones(ids.shape[1], ids.shape[1], device=ids.device, dtype=torch.bool).triu(1)
        for b in self.blocks: x = b(x, src_mask=mask, is_causal=True)
        h = self.norm(x)
        return (h if return_hidden else self.lm_head(h)), None, {}

    loss = PersistentLM.loss


def make_model(config):
    cfg = ModelConfig(**config) if isinstance(config, dict) else config
    return TransformerLM(cfg) if cfg.architecture == "transformer" else PersistentLM(cfg)
