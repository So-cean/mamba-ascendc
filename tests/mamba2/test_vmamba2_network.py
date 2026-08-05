import torch
from dataclasses import replace

from mamba_torch.vmamba2_network import (
    PureVisionMamba2,
    VisionMamba2Config,
    count_mamba_modules,
    fused_attention,
)


def test_pure_vmamba2_reference_forward():
    config = VisionMamba2Config(
        image_size=32,
        num_classes=10,
        dims=(64,),
        depths=(1,),
        ngroups=(1,),
        chunk_sizes=(64,),
        directions=1,
        dstate=64,
    )
    torch.manual_seed(7)
    model = PureVisionMamba2(config, backend="reference").eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(image)
    assert output.shape == (1, 10)
    assert torch.isfinite(output).all()
    assert count_mamba_modules(model) == {"mixer_blocks": 1, "ssd_cores": 1}


def test_sdpa_adapter_cpu():
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    output = fused_attention(q, k, v)
    assert output.shape == q.shape
    assert torch.isfinite(output).all()


def test_streaming_cross_scan_matches_stack_merge():
    config = VisionMamba2Config(
        image_size=32,
        num_classes=10,
        dims=(64,),
        depths=(1,),
        ngroups=(1,),
        chunk_sizes=(64,),
        directions=4,
        dstate=64,
        streaming_merge=False,
    )
    torch.manual_seed(11)
    stack_model = PureVisionMamba2(config, backend="reference").eval()
    stream_model = PureVisionMamba2(
        replace(config, streaming_merge=True), backend="reference"
    ).eval()
    stream_model.load_state_dict(stack_model.state_dict())
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = stack_model(image)
        actual = stream_model(image)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
