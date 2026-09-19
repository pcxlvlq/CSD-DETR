import os
from pathlib import Path
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent

MODEL_CFG = ROOT / "configs" / "rtdetr-CSP-MSLCB-CSSA-RepC3_DE.yaml"
DATA_CFG = ROOT / "configs" / "ltx_det.yaml"

model = RTDETR(str(MODEL_CFG))

results = model.train(
    data=str(DATA_CFG),
    imgsz=640,
    epochs=200,
    batch=16,
    device=[0, 1, 2, 3],
    workers=12,
    optimizer="AdamW",
    lr0=0.0001,
    lrf=1,
    momentum=0.9,
    weight_decay=0.0001,
)

metrics = model.val(
    data=str(DATA_CFG),
    split="test",
)

