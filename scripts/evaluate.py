import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model performance with model-specific output folders.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11s_v3/best.pt",
        help="Path to YOLO weights file (default: models/yolo11s_v3/best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to dataset YAML file (default: data.yaml)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output subfolder name for this model evaluation (defaults to model folder name)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/val",
        help="Base directory for validation runs (default: runs/val)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on (val or test)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for evaluation (default: 640)",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=True,
        help="Overwrite existing folder instead of incrementing",
    )
    return parser.parse_args()


def get_model_identifier(model_path: Path, custom_name: str | None) -> str:
    if custom_name:
        return custom_name
    parent_name = model_path.parent.name
    if parent_name and parent_name not in ["models", ".", ""]:
        return parent_name
    return model_path.stem


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Error: Model not found at '{model_path}'")
        return

    model_name = get_model_identifier(model_path, args.name)
    save_dir = Path(args.project) / model_name

    print("=" * 60)
    print(f" Evaluating Model:  {model_path}")
    print(f" Model Identifier:  {model_name}")
    print(f" Dataset config:    {args.data}")
    print(f" Output folder:     {save_dir}")
    print("=" * 60)

    model = YOLO(str(model_path))

    project_dir = str(Path(args.project).resolve())

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        project=project_dir,
        name=model_name,
        exist_ok=args.exist_ok,
    )

    print("\n" + "=" * 60)
    print(f" EVALUATION METRICS FOR: [{model_name}]")
    print("=" * 60)
    print(f" mAP50-95: {metrics.box.map:.4f}")
    print(f" mAP50:    {metrics.box.map50:.4f}")
    print(f" mAP75:    {metrics.box.map75:.4f}")
    print(f" Results & plots saved to: {save_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
