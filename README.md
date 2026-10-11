# E-Waste Object Detection

E-Waste detection project using YOLO (Ultralytics).

## 📁 Repository Structure

```text
ewaste-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── download_data.py       # Dataset downloader
│   ├── evaluate.py            # Model evaluation & metrics
│   └── predict.py             # Inference on image/video/webcam
├── models/
│   └── yolo11s_v3/
│       └── best.pt            # Model weights (place your file here)
└── results/
    └── yolo11s_v3/            # results.csv, args.yaml, plots, test_metrics.json
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Trained Model & Results
1. Place your trained model weights (`best.pt`) in `models/<model_variant>/`:
   ```text
   models/yolo11s_v3/best.pt
   ```
2. Place training plots, confusion matrix, `results.csv`, and `args.yaml` into:
   ```text
   results/yolo11s_v3/
   ```

### 3. Run Inference
The script automatically isolates results by model into `runs/predict/<model_name>/`:
```bash
# Default (runs yolo11s_v3 on sample_images, outputs to runs/predict/yolo11s_v3/)
py scripts/predict.py

# Test another teammate's model:
py scripts/predict.py --model models/yolov8_pcb/best.pt --source sample_images
# -> Automatically outputs to: runs/predict/yolov8_pcb/

# On a single image:
py scripts/predict.py --model models/yolo11s_v3/best.pt --source sample_images/sample_1_pcb_board.jpg

# Live webcam:
py scripts/predict.py --source 0 --show
```

### 4. Evaluate Model
Results and validation plots automatically save to `runs/val/<model_name>/`:
```bash
py scripts/evaluate.py --model models/yolo11s_v3/best.pt --data data.yaml --split val
```


---

## 👥 Team Model Registry

| Model Name | Trained By | Architecture | Classes | Dataset / Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `yolo11s_v3` | Hari | YOLO11s | 18 classes (IC, PCBboard, capacitor, etc.) | Roboflow v3 | ✅ Ready |

### Classes (18)
`IC`, `LED`, `PCBboard`, `capacitor`, `crystaloscillator`, `hole`, `inductor`, `metal`, `microcontroller`, `mousewheelencoder`, `resistor`, `resistor-capacitor`, `spring`, `switch-button`, `transformer`, `usb-type-a-connector-body`, `wire`, `wire-connector`

