from ultralytics import YOLO
import logging
import cv2
import json
import os

CONFIG_FILE = "./robot/camera_config.json"
TARGET_LABELS = {"skateboard"}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"car_scale": {}}

def save_scale(width_px, height_px):
    config = load_config()
    config["car_scale"] = {
        "width_px": width_px,
        "height_px": height_px,
        "pixels_per_car_length": width_px  # car length = the long axis of bounding box
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Car scale saved: {width_px}x{height_px}px  ({width_px} px per car length)")

def main():
    model = YOLO("yolov8n.pt")
    logging.getLogger("ultralytics").setLevel(logging.ERROR) # Prevent printing for each frame
    cap = cv2.VideoCapture(0)

    print("When you see a good bounding box, press 's' to save the scale.")
    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.1)
        best_box = None  # track the highest-conf target detection

        for box in results[0].boxes:
            cls   = int(box.cls[0])
            label = model.names[cls]
            conf  = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw all detections in red to show what's visible
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if label in TARGET_LABELS:
                if best_box is None or conf > float(best_box.conf[0]):
                    best_box = box

        # Highlight the best target box in green
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{w}x{h}px  <- press S to save",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Bounding Box Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s') and best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            save_scale(x2 - x1, y2 - y1)
            break  # done

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()