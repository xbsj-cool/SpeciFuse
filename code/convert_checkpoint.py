"""Merge three SpeciFuse checkpoints into one file.

The original training script saves three files per epoch:
- ViE_Epoch*.pth
- IrE_Epoch*.pth
- FuseD_Epoch*.pth

This script packs them into a single checkpoint consumed by ``fusion_test.py``.
"""

import argparse
from pathlib import Path
from typing import Any, Dict


MODEL_KEY = "model"
VIS_ENCODER_KEY = "vis_encoder"
IR_ENCODER_KEY = "ir_encoder"
FUSION_DECODER_KEY = "fusion_decoder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SpeciFuse checkpoint files.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("ablation_ijcai"),
        help="Directory containing the original separate checkpoint files.",
    )
    parser.add_argument("--epoch", type=int, default=60, help="Checkpoint epoch to merge.")
    parser.add_argument(
        "--vis-encoder",
        type=Path,
        default=None,
        help="Optional explicit path to ViE checkpoint.",
    )
    parser.add_argument(
        "--ir-encoder",
        type=Path,
        default=None,
        help="Optional explicit path to IrE checkpoint.",
    )
    parser.add_argument(
        "--fusion-decoder",
        type=Path,
        default=None,
        help="Optional explicit path to FuseD checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output merged checkpoint path. Defaults to weights-dir/SpeciFuse_Epoch{epoch}.pth.",
    )
    parser.add_argument(
        "--map-location",
        type=str,
        default="cpu",
        help="Device used when loading source checkpoints.",
    )
    return parser.parse_args()


def default_source_paths(weights_dir: Path, epoch: int) -> Dict[str, Path]:
    return {
        VIS_ENCODER_KEY: weights_dir / f"ViE_Epoch{epoch}.pth",
        IR_ENCODER_KEY: weights_dir / f"IrE_Epoch{epoch}.pth",
        FUSION_DECODER_KEY: weights_dir / f"FuseD_Epoch{epoch}.pth",
    }


def resolve_source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    paths = default_source_paths(args.weights_dir, args.epoch)
    if args.vis_encoder is not None:
        paths[VIS_ENCODER_KEY] = args.vis_encoder
    if args.ir_encoder is not None:
        paths[IR_ENCODER_KEY] = args.ir_encoder
    if args.fusion_decoder is not None:
        paths[FUSION_DECODER_KEY] = args.fusion_decoder
    return paths


def load_checkpoint(path: Path, map_location: str) -> Any:
    import torch

    if not path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def merge_checkpoints(
    source_paths: Dict[str, Path],
    output_path: Path,
    epoch: int,
    map_location: str,
) -> None:
    import torch

    merged_checkpoint = {
        "meta": {
            "format": "specifuse_merged_checkpoint",
            "version": 1,
            "epoch": epoch,
            "source_files": {name: str(path) for name, path in source_paths.items()},
        },
        MODEL_KEY: {
            VIS_ENCODER_KEY: load_checkpoint(source_paths[VIS_ENCODER_KEY], map_location),
            IR_ENCODER_KEY: load_checkpoint(source_paths[IR_ENCODER_KEY], map_location),
            FUSION_DECODER_KEY: load_checkpoint(source_paths[FUSION_DECODER_KEY], map_location),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_checkpoint, output_path)


def run(args: argparse.Namespace) -> None:
    source_paths = resolve_source_paths(args)
    output_path = args.output or args.weights_dir / f"SpeciFuse_Epoch{args.epoch}.pth"

    merge_checkpoints(
        source_paths=source_paths,
        output_path=output_path,
        epoch=args.epoch,
        map_location=args.map_location,
    )

    print(f"Merged checkpoint saved to: {output_path}")


if __name__ == "__main__":
    run(parse_args())
