import cv2 as cv
import numpy as np

# =========================
# Assigned Target Color
# =========================
ASSIGNED_COLOR = "blue"   # Change this to "red", "green", or "yellow" as needed

# =========================
# HSV Color Ranges
# =========================
COLOR_RANGES = {
    "blue":   [(np.array([90, 80, 40]), np.array([140, 255, 255]))],
    "red":    [(np.array([0, 80, 40]), np.array([10, 255, 255])),
               (np.array([165, 80, 40]), np.array([180, 255, 255]))],
    "green":  [(np.array([35, 60, 40]), np.array([85, 255, 255]))],
    "yellow": [(np.array([15, 80, 40]), np.array([40, 255, 255]))]
}


# =========================
# Distance Calibration
# =========================
KNOWN_DISTANCE = 150.0      # cm (distance at which you calibrate)
REAL_GATE_WIDTH = 100.0     # cm (1 meter gate width)
FOCAL_LENGTH = None         # will be calculated once

# =========================
# Camera Setup
# =========================
cap = cv.VideoCapture(0)

cv.namedWindow("Frame", cv.WINDOW_NORMAL)
cv.namedWindow("Mask", cv.WINDOW_NORMAL)
cv.namedWindow("Edges", cv.WINDOW_NORMAL)
cv.resizeWindow("Frame", 800, 600)
cv.resizeWindow("Mask", 400, 300)
cv.resizeWindow("Edges", 400, 300)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------
    # Preprocessing
    # -------------------------
    frame = cv.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # White balance / contrast enhancement
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv.merge((l,a,b))
    frame = cv.cvtColor(lab, cv.COLOR_LAB2BGR) 

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = cv.equalizeHist(v)
    hsv = cv.merge((h, s, v))

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    h, w, _ = frame.shape
    frame_cx, frame_cy = w // 2, h // 2

    detections = {}   # store all detected gates
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # -------------------------
    # Detect all colors
    # -------------------------
    for color_name, ranges in COLOR_RANGES.items():
        mask = None
        for lower, upper in ranges:
            temp_mask = cv.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else cv.bitwise_or(mask, temp_mask)

        edges = cv.Canny(gray, 50, 150)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

        combined_mask = cv.bitwise_or(combined_mask, mask)

        # Find contours for this color
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            gate = max(contours, key=cv.contourArea)
            area = cv.contourArea(gate)
            if area > 500:
                x, y, bw, bh = cv.boundingRect(gate)
                cx, cy = x + bw // 2, y + bh // 2

                # Draw gate
                cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv.putText(frame, color_name, (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                detections[color_name] = {"cx": cx, "cy": cy, "bw": bw, "bh": bh}

    # -------------------------
    # Navigation logic for assigned color
    # -------------------------
    command = "SEARCH"
    distance_cm = None

    if ASSIGNED_COLOR in detections:
        target = detections[ASSIGNED_COLOR]
        gate_cx, gate_cy, bw = target["cx"], target["cy"], target["bw"]

        cv.circle(frame, (gate_cx, gate_cy), 5, (0, 0, 255), -1)

        # Distance estimation
        if FOCAL_LENGTH is None:
            FOCAL_LENGTH = (bw * KNOWN_DISTANCE) / REAL_GATE_WIDTH
        distance_cm = (REAL_GATE_WIDTH * FOCAL_LENGTH) / bw

        cv.putText(frame, f"Distance: {int(distance_cm)} cm", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Alignment logic
        dx = gate_cx - frame_cx
        dy = gate_cy - frame_cy

        if abs(dx) > abs(dy):
            if dx > 40:
                command = "MOVE RIGHT"
            elif dx < -40:
                command = "MOVE LEFT"
            else:
                command = "CENTERED"
        else:
            if dy > 30:
                command = "MOVE UP"
            elif dy < -30:
                command = "MOVE DOWN"
            else:
                command = "CENTERED"

    # -------------------------
    # Display
    # -------------------------
    cv.line(frame, (frame_cx, 0), (frame_cx, h), (255, 255, 255), 1)
    cv.line(frame, (0, frame_cy), (w, frame_cy), (255, 255, 255), 1)

    cv.putText(frame, f"COMMAND: {command}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.imshow("Frame", frame)
    cv.imshow("Mask", combined_mask)
    cv.imshow("Edges", edges)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
