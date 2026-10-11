import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download the E-Waste dataset from Roboflow.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY environment variable)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="ai-cftyz",
        help="Roboflow workspace name (default: ai-cftyz)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="e-waste-classification-a25jm",
        help="Roboflow project name (default: e-waste-classification-a25jm)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=3,
        help="Dataset version number (default: 3)",
    )
    parser.add_argument(
        "--model-format",
        type=str,
        default="yolov11",
        help="Download format (default: yolov11)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="datasets",
        help="Destination directory (default: datasets)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        print("=" * 60)
        print("ERROR: Roboflow API key is required.")
        print("You can provide it via:")
        print("  1) Command line:  python scripts/download_data.py --api-key YOUR_API_KEY")
        print("  2) Environment:   set ROBOFLOW_API_KEY=YOUR_API_KEY")
        print("=" * 60)
        return

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow library not installed. Install it with:")
        print("  pip install roboflow")
        return

    print(f"Connecting to Roboflow workspace '{args.workspace}', project '{args.project}'...")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset (version {args.version}, format {args.model_format}) to '{dest_dir}'...")
    dataset = version.download(model_format=args.model_format, location=str(dest_dir))
    print(f"Successfully downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
