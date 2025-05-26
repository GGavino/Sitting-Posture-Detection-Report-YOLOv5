import sys
import cv2
import csv
import os
from app_models.model import Model

def draw_box(frame, bbox, label, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main(video_path, model_path="small640.pt", target_fps=5):
    model = Model(model_path, use_camera=False)
    inference_model = model.inference_model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video path
    os.makedirs("processed_videos", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"processed_videos/{base_name}_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_counts = {"sitting_good": 0, "sitting_bad": 0, "undetected": 0}
    total_frames = 0
    processed_frames = 0

    class_map = {0: "sitting_good", 1: "sitting_bad"}
    color_map = {"sitting_good": (0,255,0), "sitting_bad": (0,0,255), "undetected": (128,128,128)}

    frame_interval = max(1, int(round(fps / target_fps)))

    last_bbox = None
    last_label = "undetected"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = last_label
        bbox = last_bbox

        if total_frames % frame_interval == 0:
            results = inference_model.predict(frame)
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_idx, confidence = inference_model.get_results(results)

            if class_idx is None:
                frame_counts["undetected"] += 1
                label = "undetected"
                bbox = None
            else:
                label = class_map.get(class_idx, "undetected")
                frame_counts[label] += 1
                bbox = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            processed_frames += 1

            # Update last processed
            last_label = label
            last_bbox = bbox

        # Draw box and label if detected
        if bbox is not None:
            draw_box(frame, bbox, label, color_map[label])
        elif label == "undetected":
            cv2.putText(frame, "undetected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map[label], 2)

        out_video.write(frame)
        total_frames += 1

    cap.release()
    out_video.release()

    print("Processed video saved to:", out_path)
    print("Video FPS:", fps)
    print("Total frames:", total_frames)
    print("Processed frames:", processed_frames)
    for k, v in frame_counts.items():
        print(f"{k}: {v} frames, {v/target_fps:.2f} seconds")

    # Save report to CSV
    csv_filename = "posture_report.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')  # Use semicolon as delimiter
        writer.writerow(["Posture", "Frames", "Seconds"])
        for k, v in frame_counts.items():
            writer.writerow([k, v, f"{v/target_fps:.2f}"])
    print(f"Report saved to {csv_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <video_path> [model_path] [target_fps]")
        sys.exit(1)
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "small640.pt"
    target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    main(video_path, model_path, target_fps)