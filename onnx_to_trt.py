from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_engine.exporter import ExportEngine


def main() -> int:
    ap = argparse.ArgumentParser(description="ONNX -> TensorRT converter")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--mode", choices=["single", "batch"], default="single")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--input-shape", default="3x640x640",
                    help="CxHxW (e.g. 3x1280x1280)")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--input-name", default=None)
    args = ap.parse_args()

    c, h, w = (int(x) for x in args.input_shape.split("x"))

    exp = ExportEngine(output_dir=str(Path(args.engine).parent))
    out = exp.export_onnx_to_trt(
        onnx_path=args.onnx,
        mode=args.mode,
        max_batch=args.max_batch,
        fp16=not args.no_fp16,
        input_shape=(c, h, w),
        output_path=args.engine,
        input_name=args.input_name,
    )
    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
