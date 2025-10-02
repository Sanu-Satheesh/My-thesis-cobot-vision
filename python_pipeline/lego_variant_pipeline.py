#lego_variant_pipeline.py  THIS WILL WORK..!

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from collections import deque, Counter

# ====== CONFIG ======
MODEL_PATH = r"D:\My_thesis\best.pt"
CONF_THR = 0.35
IOU_THR  = 0.5
HIST_LEN = 10  # frames for smoothing
# ====================

# === Rule-based decisions ===
def decide_class(has_front_tyre, has_drum, has_rear_tyre, has_rear_grey):
    """
    Decide LEGO class (C1–C4) based on detected modules.
    """
    if has_front_tyre and has_rear_grey: return "C1"
    if has_front_tyre and has_rear_tyre: return "C2"
    if has_drum and has_rear_tyre:       return "C3"
    if has_drum and has_rear_grey:       return "C4"
    return None

def decide_variant(led_count):
    """
    Decide LEGO variant (V1–V4) based on LED count.
    """
    if led_count == 1: return "V1"
    elif led_count == 2: return "V2"
    elif led_count == 3: return "V3"
    elif led_count == 4: return "V4"
    return None
# =============================

def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Setup RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    # Smoothing buffer
    history = deque(maxlen=HIST_LEN)

    try:
        while True:
            # Get frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            # YOLO inference
            results = model.predict(frame, conf=CONF_THR, iou=IOU_THR, verbose=False)[0]

            # Detection flags
            has_front_tyre = has_drum = has_rear_tyre = has_rear_grey = False
            led_count = 0

            # Draw detections
            for box in results.boxes:
                cls = int(box.cls[0])
                name = results.names[cls]
                conf = float(box.conf[0])
                if conf < CONF_THR:
                    continue

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Set flags
                if   name == "front_tyre":       has_front_tyre = True
                elif name == "drum_roller":      has_drum = True
                elif name == "rear_tyre":        has_rear_tyre = True
                elif name == "rear_grey_wheels": has_rear_grey = True
                elif name == "led":              led_count += 1

            # Apply rules
            lego_class = decide_class(has_front_tyre, has_drum, has_rear_tyre, has_rear_grey)
            lego_variant = decide_variant(led_count) if lego_class else None

            tag = f"{lego_class or 'C?'}-{lego_variant or 'V?'}"
            history.append(tag)

            # Stable decision (majority vote over last N frames)
            stable = Counter(history).most_common(1)[0][0]

            # Display HUD
            cv2.putText(frame, f"RAW: {tag} (LEDs={led_count})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"STABLE: {stable}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("LEGO Variant Detector", frame)

            # 🚀 Hook: send `stable` to your UR3e cobot here
            # e.g., write to file/socket/PLC depending on integration

            # Quit
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or Q
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
