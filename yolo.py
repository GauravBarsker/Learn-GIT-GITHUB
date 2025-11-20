import cv2
import time
from ultralytics import YOLO

# ---------------------------------------------------
# LOAD BEST MODEL FOR HEAVY CROWD DETECTION
# ---------------------------------------------------
# Make sure yolov8l-crowdhuman.pt is in same folder
model = YOLO("yolov8l-crowdhuman.pt")

# ---------------------------------------------------
# SET CAMERA INDEX (CHANGE IF NEEDED)
# ---------------------------------------------------
# 0 = laptop webcam
# 1 = DroidCam (most likely)
# 2 or 3 = if multiple cameras exist
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera. Check device index or DroidCam.")
    exit()

# ---------------------------------------------------
# TRACKING + COUNTING VARIABLES
# ---------------------------------------------------
total_ids = set()        # stores ALL unique IDs detected ever
id_last_seen = {}        # timestamp of last time ID was seen
MEMORY_TIME = 5          # seconds (avoid double counting)

print("🚀 YOLOv8 CrowdHuman Live Counting Started (Press 'q' to quit)...")

# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera read error! Check connection.")
        break

    current_time = time.time()
    frame_count = 0
    active_ids = set()

    # Run YOLO + ByteTrack tracking
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes is not None:
        for box in results[0].boxes:

            # Only detect PERSON class
            cls = int(box.cls[0])
            if cls != 0:
                continue  # Skip anything that's not a person

            tid = int(box.id[0]) if box.id is not None else None

            # If tracking ID exists
            if tid is not None:
                active_ids.add(tid)
                id_last_seen[tid] = current_time

                # Count once per unique ID
                if tid not in total_ids:
                    total_ids.add(tid)

            frame_count += 1

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Put ID on screen
            if tid is not None:
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ---------------------------------------------------
    # CLEAN OLD IDs (so same person not counted twice)
    # ---------------------------------------------------
    ids_to_remove = []
    for tid in list(id_last_seen.keys()):
        if current_time - id_last_seen[tid] > MEMORY_TIME:
            ids_to_remove.append(tid)

    for tid in ids_to_remove:
        del id_last_seen[tid]
        # DO NOT remove from total_ids → prevents double-counting

    # ---------------------------------------------------
    # TEXT OVERLAYS
    # ---------------------------------------------------
    cv2.putText(frame, f"Frame Count: {frame_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.putText(frame, f"Total Count: {len(total_ids)}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # ---------------------------------------------------
    # SHOW OUTPUT
    # ---------------------------------------------------
    cv2.imshow("YOLOv8-L CrowdHuman Live (DroidCam)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n🎉 Session Ended")
print("👥 Total Unique Persons Detected:", len(total_ids))
