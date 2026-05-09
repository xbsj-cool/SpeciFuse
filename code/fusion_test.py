"""Run visible/infrared image fusion on a test set.

Defaults mirror the original test script:
- visible images are read from ``Zero-DCE_code/fuse_data/test_data/vi_h``
- infrared paths are inferred by replacing ``vi`` with ``ir``
- checkpoint is loaded from ``ablation_ijcai/SpeciFuse_Epoch60.pth``
- output paths are inferred by replacing ``test_data`` with ``result``
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MODEL_KEY = "model"
VIS_ENCODER_KEY = "vis_encoder"
IR_ENCODER_KEY = "ir_encoder"
FUSION_DECODER_KEY = "fusion_decoder"


@dataclass
class FusionModels:
    vis_encoder: Any
    ir_encoder: Any
    fusion_decoder: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SpeciFuse image fusion tests.")
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=Path("fuse_data/test_data/vi"),
        help="Visible-image file or directory.",
    )
    parser.add_argument(
        "--ir-dir",
        type=Path,
        default=Path("fuse_data/test_data/ir"),
        help="Optional infrared root directory. If omitted, paths are inferred by text replacement.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("fuse_data/result"),
        help="Optional output root directory. If omitted, the original result path rule is used.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint.pth"),
        help="Merged checkpoint produced by convert_checkpoint.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string, for example cuda:0 or cpu. Defaults to cuda:0 when available.",
    )
    parser.add_argument(
        "--vis-token",
        type=str,
        default="vi",
        help="Token replaced when inferring infrared paths.",
    )
    parser.add_argument(
        "--ir-token",
        type=str,
        default="ir",
        help="Replacement token used when inferring infrared paths.",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Run modules in eval mode. Disabled by default to preserve the original script behavior.",
    )
    parser.add_argument(
        "--load-once",
        action="store_true",
        help="Load checkpoint once for the whole run. The original script reloads it per image.",
    )
    return parser.parse_args()


def resolve_device(device_name: Optional[str]) -> Any:
    import torch

    if device_name:
        return torch.device(device_name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_merged_checkpoint(path: Path, device: Any) -> dict:
    import torch

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def get_model_state_dicts(checkpoint: dict) -> dict:
    if MODEL_KEY in checkpoint:
        checkpoint = checkpoint[MODEL_KEY]

    required_keys = [VIS_ENCODER_KEY, IR_ENCODER_KEY, FUSION_DECODER_KEY]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(
            "Merged checkpoint is missing model keys: "
            + ", ".join(missing_keys)
            + ". Please regenerate it with code/convert_checkpoint.py."
        )

    return checkpoint


def load_models(
    checkpoint_path: Path,
    device: Any,
    eval_mode: bool,
) -> FusionModels:
    from fusion_model import Encoder, FusionDecoderWithDegradation

    checkpoint = load_merged_checkpoint(checkpoint_path, device)
    state_dicts = get_model_state_dicts(checkpoint)

    vis_encoder = Encoder().to(device)
    ir_encoder = Encoder().to(device)
    fusion_decoder = FusionDecoderWithDegradation(128, 128).to(device)

    vis_encoder.load_state_dict(state_dicts[VIS_ENCODER_KEY])
    ir_encoder.load_state_dict(state_dicts[IR_ENCODER_KEY])
    fusion_decoder.load_state_dict(state_dicts[FUSION_DECODER_KEY])

    if eval_mode:
        vis_encoder.eval()
        ir_encoder.eval()
        fusion_decoder.eval()

    return FusionModels(
        vis_encoder=vis_encoder,
        ir_encoder=ir_encoder,
        fusion_decoder=fusion_decoder,
    )


def iter_image_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
        return

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            yield candidate


def read_image_tensor(path: Path, device: Any) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    tensor = torch.from_numpy(image_array).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def infer_ir_path(
    vis_path: Path,
    vis_root: Path,
    ir_root: Optional[Path],
    vis_token: str,
    ir_token: str,
) -> Path:
    if ir_root is not None:
        try:
            return ir_root / vis_path.relative_to(vis_root)
        except ValueError:
            return ir_root / vis_path.name

    return Path(str(vis_path).replace(vis_token, ir_token))


def infer_result_path(
    vis_path: Path,
    vis_root: Path,
    result_root: Optional[Path],
) -> Path:
    if result_root is not None:
        try:
            return result_root / vis_path.relative_to(vis_root)
        except ValueError:
            return result_root / vis_path.name

    original_style_path = vis_path.as_posix()
    original_style_path = original_style_path.replace("test_data", "result")
    original_style_path = original_style_path.replace("vi/", "")
    return Path(original_style_path)


def fuse_image(
    vis_path: Path,
    vis_root: Path,
    ir_root: Optional[Path],
    result_root: Optional[Path],
    checkpoint_path: Path,
    device: Any,
    vis_token: str,
    ir_token: str,
    eval_mode: bool,
    models: Optional[FusionModels] = None,
) -> Tuple[Path, float]:
    from torchvision.utils import save_image

    ir_path = infer_ir_path(vis_path, vis_root, ir_root, vis_token, ir_token)
    result_path = infer_result_path(vis_path, vis_root, result_root)

    if not ir_path.exists():
        raise FileNotFoundError(f"Infrared image not found for {vis_path}: {ir_path}")

    vis = read_image_tensor(vis_path, device)
    ir = read_image_tensor(ir_path, device)
    current_models = models or load_models(checkpoint_path, device, eval_mode)

    start_time = time.time()
    features_vis, _, attention_vis = current_models.vis_encoder(vis)
    features_ir, _, attention_ir = current_models.ir_encoder(ir)
    fused = current_models.fusion_decoder(
        features_vis,
        features_ir,
        attention_vis,
        attention_ir,
    )
    elapsed = time.time() - start_time

    result_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(fused, result_path)
    return result_path, elapsed


def run(args: argparse.Namespace) -> None:
    import torch

    device = resolve_device(args.device)
    vis_root = args.vis_dir.parent if args.vis_dir.is_file() else args.vis_dir
    image_paths = list(iter_image_paths(args.vis_dir))

    if not image_paths:
        raise FileNotFoundError(f"No test images found under: {args.vis_dir}")

    cached_models = None
    if args.load_once:
        cached_models = load_models(args.checkpoint, device, args.eval_mode)

    total_time = 0.0
    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            result_path, elapsed = fuse_image(
                vis_path=image_path,
                vis_root=vis_root,
                ir_root=args.ir_dir,
                result_root=args.result_dir,
                checkpoint_path=args.checkpoint,
                device=device,
                vis_token=args.vis_token,
                ir_token=args.ir_token,
                eval_mode=args.eval_mode,
                models=cached_models,
            )
            total_time += elapsed
            print(f"[{index}/{len(image_paths)}] {image_path} -> {result_path} ({elapsed:.4f}s)")

    print(f"Total inference time: {total_time:.4f}s")


if __name__ == "__main__":
    run(parse_args())
