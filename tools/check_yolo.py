from ultralytics import YOLO
import os
candidates = [
    os.path.join(os.path.dirname(__file__), '..', p)
    for p in ('yolo11n.pt', 'yolo11n-seg.pt', 'yolo26n-seg.pt')
]
for c in candidates:
    c = os.path.abspath(c)
    if os.path.exists(c):
        print('Found weights at', c)
        try:
            m = YOLO(c)
            print('Model loaded. Class names:')
            print(m.names)
        except Exception as e:
            print('Failed to load model:', e)
        break
else:
    print('No local YOLO weights found at expected paths:', candidates)
