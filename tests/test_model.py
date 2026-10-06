from dataclasses import replace
import torch
import pytest
from tfplslm.model import ModelConfig, PersistentLM, detach_state


def tiny(backend="reference"):
    return ModelConfig(vocab_size=64, d_model=32, layers=2, d_state=16, expand=2,
                       head_dim=16, ff_dim=64, scan_chunk=32, memory_slots=4,
                       memory_dim=8, memory_chunk=32, backend=backend)


def test_causality_and_stream_equivalence():
    torch.manual_seed(7)
    model = PersistentLM(tiny()).eval()
    ids = torch.randint(0, 64, (2, 32))
    whole, state, _ = model(ids)
    other = ids.clone(); other[:, 20:] = torch.randint(0, 64, (2, 12))
    changed, _, _ = model(other)
    torch.testing.assert_close(whole[:, :20], changed[:, :20])
    parts, st = [], None
    for i in range(32):
        logits, st, _ = model(ids[:, i:i+1], st)
        parts.append(logits)
    torch.testing.assert_close(torch.cat(parts, 1), whole, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(st["memory"], state["memory"], atol=1e-5, rtol=1e-4)
    assert st["position"] == 0


def test_cross_chunk_write_gradient_and_reset():
    torch.manual_seed(11)
    model = PersistentLM(tiny())
    ids = torch.randint(0, 64, (2, 64))
    _, state, _ = model(ids[:, :32])
    logits, state2, _ = model(ids[:, 32:], state)
    logits.square().mean().backward()
    assert model.memory.write.weight.grad.abs().sum() > 0
    assert model.blocks[0].proj.weight.grad.isfinite().all()
    fresh, _, _ = model(ids[:, 32:])
    reset, _, _ = model(ids[:, 32:], model.initial_state(2))
    torch.testing.assert_close(fresh, reset)
    assert (logits - reset).abs().max() > 1e-5
    assert not detach_state(state2)["memory"].requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_matches_reference_and_gradients():
    torch.manual_seed(19)
    cfg = tiny("triton")
    gpu = PersistentLM(cfg).cuda()
    ref = PersistentLM(replace(cfg, backend="reference")).cuda()
    ref.load_state_dict(gpu.state_dict())
    ids = torch.randint(0, 64, (2, 64), device="cuda")
    outputs = []
    for model in [gpu, ref]:
        _, s, _ = model(ids[:, :32])
        y, s, _ = model(ids[:, 32:], s)
        y.square().mean().backward()
        outputs.append((y, s))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=3e-4, rtol=5e-3)
    for (name, a), (_, b) in zip(gpu.named_parameters(), ref.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-4, rtol=4e-2, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_two_chunks_backward():
    model = PersistentLM(tiny("triton")).cuda()
    ids = torch.randint(0, 64, (2, 64), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, s, _ = model(ids[:, :32])
        y, s, _ = model(ids[:, 32:], s)
        loss = y.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.memory.write.weight.grad.abs().sum() > 0
