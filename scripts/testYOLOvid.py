import cv2
import time
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8-Objekterkennung auf Video"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Pfad zur Eingabe-Videodatei (z.B. input.mp4)"
    )
    parser.add_argument(
        "-o", "--output", default="output.mp4",
        help="Pfad zur Ausgabe-Videodatei (Standard: output.mp4)"
    )
    parser.add_argument(
        "-f", "--fps", type=float, default=None,
        help="FPS für das Ausgabe-Video (Standard: Input-FPS)"
    )
    parser.add_argument(
        "-m", "--model", default="../detModels/yolov8n+SoccerNet5class_phase2/weights/best.pt",
        help="YOLO-Gewichtsdatei (Standard: best.pt)"
    )
    args = parser.parse_args()

    # Video öffnen
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Fehler: Konnte Video nicht öffnen: {args.input}")
        return

    # Parameter auslesen
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = args.fps if args.fps is not None else input_fps

    # VideoWriter einrichten
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (width, height))

    # YOLO-Modell laden
    model = YOLO(args.model)

    # Farb-Mapping für Klassen
    color_map = {
        'player':        (0, 255, 0),     # Grün
        'goalkeeper':    (255, 0, 0),     # Blau
        'ball':          (0, 165, 255),   # Orange
        'main referee':  (0, 0, 255),     # Rot
        'side referee':  (255, 255, 0),   # Cyan
        'other':         (128, 128, 128), # Grau
    }

    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferenz
        results = model(frame)[0]

        # Annotieren
        for box in results.boxes:
            cls_id = int(box.cls)
            label  = model.names[cls_id]
            conf   = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = color_map.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        print(f"Frame {frame_idx}")
        writer.write(frame)
        frame_idx += 1

    end_time = time.time()
    duration = end_time - start_time
    print(f"Fertig! {frame_idx} Frames in {duration:.2f} s verarbeitet.")
    print(f"Video gespeichert unter: {args.output}")

    cap.release()
    writer.release()

if __name__ == "__main__":
    main()
