import sys
import cv2
import csv
from app_models.model import Model

def main(video_path, model_path="small640.pt", target_fps=5):
    model = Model(model_path, use_camera=False)
    inference_model = model.inference_model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_counts = {"sitting_good": 0, "sitting_bad": 0, "undetected": 0}
    total_frames = 0
    processed_frames = 0

    # Map class indices to names (adjust if your model uses different indices)
    class_map = {0: "sitting_good", 1: "sitting_bad"}

    # Calculate frame interval
    frame_interval = max(1, int(round(fps / target_fps)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if total_frames % frame_interval == 0:
            results = inference_model.predict(frame)
            _, _, _, _, class_idx, confidence = inference_model.get_results(results)

            if class_idx is None:
                frame_counts["undetected"] += 1
            else:
                class_name = class_map.get(class_idx, "undetected")
                frame_counts[class_name] += 1
            processed_frames += 1

        total_frames += 1

    cap.release()

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