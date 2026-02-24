from ultralytics import YOLO
import os
import pandas as pd

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Folder containing 2000 images
image_folder = "dataset/images/train"  # change if needed

results_data = []

# Process entire folder at once (MUCH faster than looping manually)
results = model.predict(
    source=image_folder,
    conf=0.3,
    imgsz=320,
    save=False,
    verbose=False
)

for result in results:
    image_name = os.path.basename(result.path)

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[class_id]

        results_data.append([
            image_name,
            label,
            round(confidence, 2)
        ])

df = pd.DataFrame(results_data, columns=["Image", "Class", "Confidence"])

df.to_csv("full_batch_results.csv", index=False)

print("Processing completed. Results saved to full_batch_results.csv")
