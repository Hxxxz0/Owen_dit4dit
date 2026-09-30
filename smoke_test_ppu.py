from __future__ import annotations

import argparse

import torch


def validate_ppu() -> None:
    assert torch.cuda.is_available(), "PPU CUDA-compatible backend is unavailable"
    assert torch.cuda.device_count() > 0, "No PPU devices detected"
    device_name = torch.cuda.get_device_name(0)
    assert "PPU" in device_name, f"Expected a PPU device, got {device_name!r}"

    lhs = torch.randn(512, 512, device="cuda")
    rhs = torch.randn(512, 512, device="cuda")
    result = lhs @ rhs
    torch.cuda.synchronize()
    assert result.is_cuda and torch.isfinite(result).all().item()
    print(f"torch={torch.__version__} backend={torch.version.cuda} device={device_name}")


def validate_dit4dit() -> None:
    from DiT4DiT.model.modules.action_model.ActionDiT import ActionEncoder

    model = ActionEncoder(action_dim=7, hidden_size=64).cuda().eval()
    actions = torch.randn(2, 8, 7, device="cuda")
    timesteps = torch.rand(2, device="cuda")
    with torch.inference_mode():
        output = model(actions, timesteps)
    torch.cuda.synchronize()
    assert output.shape == (2, 8, 64) and output.is_cuda
    assert torch.isfinite(output).all().item()
    print(f"DiT4DiT ActionEncoder output={tuple(output.shape)} device={output.device}")


def validate_starvla() -> None:
    from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead

    model = L1RegressionActionHead(
        input_dim=64,
        hidden_dim=128,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
    ).cuda().eval()
    hidden = torch.randn(2, 8, 64, device="cuda")
    with torch.inference_mode():
        output = model(hidden)
    torch.cuda.synchronize()
    assert output.shape == (2, 8, 7) and output.is_cuda
    assert torch.isfinite(output).all().item()
    print(f"StarVLA action head output={tuple(output.shape)} device={output.device}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", choices=("dit4dit", "starvla"))
    args = parser.parse_args()
    validate_ppu()
    if args.project == "dit4dit":
        validate_dit4dit()
    else:
        validate_starvla()


if __name__ == "__main__":
    main()
