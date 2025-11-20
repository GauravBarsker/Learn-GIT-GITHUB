import cv2
import time
from ultralytics import YOLO

# -----------------------------
# LOAD YOLO MODEL (COCO)
# -----------------------------
# COCO-trained → person, bag, bottle, etc. sab aayega
model = YOLO("yolov8l.pt")

# -----------------------------
# CAMERA INDEX SET KARO
# -----------------------------
# 0 = laptop webcam, 1 = DroidCam (most common), 2/3 = extra cameras
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera. Check camera index or DroidCam.")
    exit()

# -----------------------------
# PERSON COUNTING VARIABLES
# -----------------------------
total_person_ids = set()      # total unique person IDs
person_last_seen = {}

# -----------------------------
# SUSPICIOUS OBJECTS (bag, bottle, etc.)
# COCO IDs:
# 24 = backpack, 26 = handbag, 28 = suitcase, 39 = bottle
# -----------------------------
SUSPICIOUS_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle"
}

obj_position = {}         # {id: (cx, cy)}
obj_stationary_time = {}  # {id: seconds stationary}
obj_last_seen = {}        # {id: last time seen}

# TEST ke liye chhota threshold rakha hai (10 sec)
# baad me isko 8*60 (8 min) kar sakte ho
UNATTENDED_LIMIT_SECONDS = 10

print("🚀 YOLOv8 Live People + Object Monitoring Started (Press 'q' to quit)...")

# -----------------------------
# MAIN LOOP
# -----------------------------
start_time_dict = {}  # for stationary timing using time, not frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera read error. Stopping.")
        break

    now = time.time()
    frame_person_count = 0

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.35,      # lower → more sensitive
        iou=0.45,
        verbose=False
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        # No detections in this frame
        cv2.putText(frame, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            tid = int(box.id[0]) if box.id is not None else None

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ---------------- PERSON LOGIC ----------------
            if cls_id == 0 and tid is not None:   # 0 = person
                frame_person_count += 1

                # unique count
                if tid not in total_person_ids:
                    total_person_ids.add(tid)
                    print(f"👤 New person detected, ID={tid}. Total unique = {len(total_person_ids)}")

                person_last_seen[tid] = now

                # draw person box (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # -------------- OBJECT (BAG/BOTTLE) LOGIC --------------
            if cls_id in SUSPICIOUS_CLASSES and tid is not None:
                label = SUSPICIOUS_CLASSES[cls_id]

                # First time seeing this object
                if tid not in obj_position:
                    obj_position[tid] = (cx, cy)
                    obj_last_seen[tid] = now
                    start_time_dict[tid] = now
                    obj_stationary_time[tid] = 0
                    print(f"📦 New {label} detected, ID={tid} at {obj_position[tid]}")

                else:
                    old_cx, old_cy = obj_position[tid]
                    movement = abs(cx - old_cx) + abs(cy - old_cy)

                    # If object barely moved → treat as stationary
                    if movement < 10:
                        obj_stationary_time[tid] = now - start_time_dict[tid]
                    else:
                        # Reset timer and update position if moved
                        obj_position[tid] = (cx, cy)
                        start_time_dict[tid] = now
                        obj_stationary_time[tid] = 0

                    obj_last_seen[tid] = now

                # Draw object box (blue)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} ID:{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # ALERT: stationary for too long
                if obj_stationary_time[tid] >= UNATTENDED_LIMIT_SECONDS:
                    print("\n🚨 UNATTENDED OBJECT ALERT 🚨")
                    print(f"Type: {label}")
                    print(f"Object ID: {tid}")
                    print(f"Location (approx pixels): {obj_position[tid]}")
                    print(f"Stationary for: {int(obj_stationary_time[tid])} seconds")
                    print("--------------------------------------")
                    # After alert, reset timer so it doesn't spam
                    start_time_dict[tid] = now
                    obj_stationary_time[tid] = 0

    # ---------------- OVERLAY TEXT ----------------
    cv2.putText(frame, f"People in Frame: {frame_person_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"Total Unique Persons: {len(total_person_ids)}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # SHOW WINDOW
    cv2.imshow("YOLOv8 Person + Object Monitor", frame)

    # QUIT KEY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()

print("\n🎉 Session Ended")
print("👥 Final Total Unique Persons Detected:", len(total_person_ids))
