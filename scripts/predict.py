import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference with model-specific output folders.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11s_v3/best.pt",
        help="Path to YOLO weights file (default: models/yolo11s_v3/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="sample_images",
        help="Source path: image file, directory, video file, or '0' for webcam (default: sample_images)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold. Default: 0.50 for live webcam, 0.25 for images.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Filter specific classes by name or index (e.g. --classes PCBboard IC microcontroller)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output subfolder name for this model (defaults to model folder, e.g. yolo11s_v3)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/predict",
        help="Base directory for predictions (default: runs/predict)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=None,
        help="Save annotated prediction images/videos (auto-enabled for files, disabled for live webcam)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live results window",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=True,
        help="Overwrite existing model prediction folder instead of creating incremental folders",
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
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please place your model inside the directory, e.g. models/yolo11s_v3/best.pt")
        return

    # Check if source is a live webcam
    is_webcam = str(args.source).isdigit() or args.source == "0"
    source = int(args.source) if is_webcam else args.source

    # Set smart defaults for confidence and saving
    if args.conf is None:
        # Webcam needs higher confidence (0.50) to suppress false background detections
        conf_thresh = 0.50 if is_webcam else 0.25
    else:
        conf_thresh = args.conf

    # By default, don't save thousands of frames from webcam to disk unless requested
    save_outputs = args.save if args.save is not None else (not is_webcam)
    show_window = args.show or is_webcam

    model_name = get_model_identifier(model_path, args.name)
    save_dir = Path(args.project) / model_name
    project_dir = str(Path(args.project).resolve())

    print("=" * 65)
    print(f" Model:            {model_path}")
    print(f" Model Identifier: {model_name}")
    print(f" Source:           {args.source} {'(Live Webcam)' if is_webcam else ''}")
    print(f" Confidence Thresh:{conf_thresh:.2f}" + (" (raised for webcam)" if is_webcam and args.conf is None else ""))
    print(f" IoU / NMS:        {args.iou:.2f}")
    if is_webcam:
        print(" Tip: Hold e-waste close to the camera with good lighting.")
        print(" Press 'q' on the camera window to exit.")
    print("=" * 65)

    model = YOLO(str(model_path))

    # Parse class filtering if provided
    filter_classes = None
    if args.classes:
        filter_classes = []
        name_to_id = {name.lower(): idx for idx, name in model.names.items()}
        for c in args.classes:
            if c.isdigit():
                filter_classes.append(int(c))
            elif c.lower() in name_to_id:
                filter_classes.append(name_to_id[c.lower()])
            else:
                print(f"Warning: Class '{c}' not recognized in model names: {list(model.names.values())}")
        if not filter_classes:
            filter_classes = None
        else:
            print(f" Filtering for classes: {[model.names[i] for i in filter_classes]}")

    results = model.predict(
        source=source,
        conf=conf_thresh,
        iou=args.iou,
        classes=filter_classes,
        imgsz=args.imgsz,
        save=save_outputs,
        show=show_window,
        project=project_dir,
        name=model_name,
        exist_ok=args.exist_ok,
        stream=is_webcam,
    )

    if is_webcam:
        # Stream results in generator
        try:
            for r in results:
                pass
        except KeyboardInterrupt:
            print("\nStopped webcam stream.")
        return

    # Static file predictions summary
    print("\n" + "=" * 65)
    print(f" DETECTION SUMMARY FOR MODEL: [{model_name}]")
    print("=" * 65)

    for i, r in enumerate(results):
        img_name = Path(r.path).name if hasattr(r, "path") else f"Item {i+1}"
        box_count = len(r.boxes)
        print(f"\n[{img_name}] -> {box_count} detection(s):")

        if box_count == 0:
            print("  (No objects detected above confidence threshold)")
            continue

        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf_val = float(box.conf[0])
            print(f"   • {cls_name:<25} (conf: {conf_val:6.1%})")

    if save_outputs:
        print("\n" + "-" * 65)
        print(f" Annotated images saved to: {save_dir.resolve()}")
        print("-" * 65)


if __name__ == "__main__":
    main()
