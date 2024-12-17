from ultralytics import YOLO;
import gc
model = YOLO('yolov8x.pt')

def clear_cache():
   gc.collect()

results = model.predict('input/frisbee_analysis.mp4', save=True)
print(results[0])
print("======================")
for box in results[0].boxes:
   print(box)
clear_cache() 