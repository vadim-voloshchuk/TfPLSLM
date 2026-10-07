import torch
from tfplslm.model import ModelConfig, make_model


def test_transformer_baseline_causal_forward_backward():
    cfg=ModelConfig(vocab_size=64,d_model=32,layers=2,ff_dim=64,memory_chunk=16,
                    memory_slots=0,architecture='transformer',backend='reference')
    model=make_model(cfg)
    ids=torch.randint(4,64,(2,16))
    logits,_,_=model(ids)
    changed=ids.clone();changed[:,8:]=torch.randint(4,64,(2,8))
    other,_,_=model(changed)
    torch.testing.assert_close(logits[:,:8],other[:,:8])
    logits.square().mean().backward()
    assert model.embedding.weight.grad.isfinite().all()
