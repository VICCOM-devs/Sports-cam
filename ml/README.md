# AnySport – model training (free pipeline)

Everything here is free: data (MIT/CC-BY), models (Apache-2.0), compute (Colab/Kaggle free tier or
your own GPU). Nothing from Ultralytics (AGPL) is used.

## What you get

`anysport-hockey.onnx`: a detector with classes
`player, goaltender, referee, puck, goal, faceoff_dot, center_ice`. Drop it in the hub's models
folder (`%LOCALAPPDATA%\VICCOM\AnySport\models\custom.onnx`) and the hub uses it instead of the
generic person detector: pucks get tracked, goalies and refs are separate classes, and faceoff
dots / goals seed the calibration.

## Steps

1. **Get data** (run on your PC or in Colab):
   ```
   python ml/prepare_data.py --out ml/data
   ```
   Downloads HockeyAI (SimulaMet, 2,101 SHL frames, MIT), unpacks it, and converts to COCO
   JSON with a 90/10 train/val split. Add your own labelled frames from the hub's labelling
   tool with `--extra <folder>` (YOLO format, same class ids).
2. **Train** – open `ml/train_colab.ipynb` in Google Colab (free T4) or run locally:
   ```
   python ml/train.py --data ml/data --model yolox-s --epochs 60 --out ml/runs/yolox-s
   ```
   `--model rtdetr-r18` trains RT-DETR instead (better on the tiny puck, heavier on CPU).
3. **Export** to ONNX for the hub:
   ```
   python ml/export.py --run ml/runs/yolox-s --out anysport-hockey.onnx
   ```
4. Copy the .onnx to the hub's models folder and restart the hub. `/api/version` shows
   `"model": "custom"`.

## Class ids (keep these fixed across datasets)

| id | class | HockeyAI name |
|---|---|---|
| 0 | player | Players |
| 1 | goaltender | Goaltender |
| 2 | referee | Referee |
| 3 | puck | Puck |
| 4 | goal | Goal Frame |
| 5 | faceoff_dot | Faceoff Dots |
| 6 | center_ice | Center Ice |

## Your own data matters most

Public data is broadcast footage from above. Phones at the glass look different. After each pilot
game, open `http://localhost:8080/label.html` on the hub, step through frames, fix boxes, export.
A few hundred corrected frames from 2–3 games is enough to adapt the model to your rink.
