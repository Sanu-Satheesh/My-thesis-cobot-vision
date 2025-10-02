# My-thesis-cobot-vision
UR3e + YOLOv8 thesis: Python pipeline, URP waypoints, Colab training

This repository contains:
- `python_pipeline/` — Python inference + UR3e control (YOLOv8 + RTDE + RealSense D455)
- `ur3e_waypoints/` — Manual programs from the UR3e teach pendant (`.urp`)
- `YOLOv8/` — Google Colab training notebooks

## Quickstart
```bash
pip install -r python_pipeline/requirements.txt
python python_pipeline/lego_variant_pipeline.py

## Hardware

UR3e (S/N 20185300461), Robotiq 2F gripper, Intel RealSense D455

Notes

Large files (e.g., .pt weights) are ignored by default.

Tested on Windows 11 + Python 3.10.
