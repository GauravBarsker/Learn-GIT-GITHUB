import cv2
import time
from ultralytics import YOLO
import numpy as np

# ---------------------------------------------------
# LOAD BEST MODEL FOR CROWD + OBJECT DETECTION
# ---------------------------------------------------
model = YOLO("yolov8l-crowdhuman.pt")

# COCO BAG CLASS IDS
BAG_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}

# ---------------------------------------------------
# SET CAMERA INDEX
# ---------------------------------------------------
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera. Check index or DroidCam.")
    exit()

# ---------------------------------------------------
# PERSON TRACKING
# ---------------------------------------------------
total_person_ids = set()
person_last_seen = {}
MEMORY_TIME = 5   # seconds

# ---------------------------------------------------
# BAG TRACKING
# ---------------------------------------------------
bag_position = {}         # {bag_id: (cx, cy)}
bag_start_time = {}       # when bag first placed
bag_stationary_time = {}  # how long bag stayed
bag_last_seen = {}        # last frame time
UNATTENDED_LIMIT = 8 * 60   # 8 minutes

print("🚀 YOLOv8 CrowdHuman + Bag Monitoring Started (Press 'q' to quit)...")

# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera read error.")
        break

    current_time = time.time()
    frame_count = 0

    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes is not None:

        for box in results[0].boxes:
            cls = int(box.cls[0])
            tid = int(box.id[0]) if box.id is not None else None

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ------------------------------
            # PERSON LOGIC
            # ------------------------------
            if cls == 0:  # person
                frame_count += 1
                person_last_seen[tid] = current_time

                if tid not in total_person_ids:
                    total_person_ids.add(tid)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Person {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # ------------------------------
            # BAG LOGIC (backpack, handbag, suitcase)
            # ------------------------------
            if cls in BAG_CLASSES:
                label = BAG_CLASSES[cls]

                # update last seen
                bag_last_seen[tid] = current_time

                # If first time seeing bag
                if tid not in bag_position:
                    bag_position[tid] = (cx, cy)
                    bag_start_time[tid] = current_time
                    bag_stationary_time[tid] = 0

                else:
                    old_cx, old_cy = bag_position[tid]
                    movement = abs(cx - old_cx) + abs(cy - old_cy)

                    # If bag hasn't moved
                    if movement < 10:
                        bag_stationary_time[tid] += 1
                    else:
                        # If moved → reset
                        bag_stage = (cx, cy)
                        bag_start_time[tid] = current_time
                        bag_stationary_time[tid] = 0
                        bag_position[tid] = (cx, cy)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame, f"Bag {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                # ALERT if bag is stationary for > 8 minutes
                if bag_stationary_time[tid] >= UNATTENDED_LIMIT:
                    print("\n🚨 UNATTENDED BAG ALERT 🚨")
                    print(f"Bag ID: {tid}")
                    print(f"Type: {label}")
                    print("Location:", bag_position[tid])
                    print("Time Left: 8+ minutes")
                    print("----------------------------------")

    # ------------------------------
    # TEXT OVERLAY
    # ------------------------------
    cv2.putText(frame, f"People in Frame: {frame_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    cv2.putText(frame, f"Total Unique Persons: {len(total_person_ids)}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # SHOW WINDOW
    cv2.imshow("YOLOv8 Live + Unattended Bag Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
