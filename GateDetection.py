import cv2 as cv
import numpy as np


ASSIGNED_COLOR = "blue"   # Change the color here

# =========================
# HSV Color Range
# =========================
COLOR_RANGES = {
    "blue": [
        (np.array([100, 120, 50]), np.array([140, 255, 255]))
    ],
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "green": [
        (np.array([40, 70, 70]), np.array([80, 255, 255]))
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255]))
    ]
}

# =========================
# Camera
# =========================


cap = cv.VideoCapture(0)      

# Incase there is window sizing issue, uncomment the code right below

cv.namedWindow("Frame", cv.WINDOW_NORMAL)   # Fixes window size issue for window named 'Frame'
cv.namedWindow("Mask", cv.WINDOW_NORMAL)    # Fixes window size issue for window named 'Mask'
cv.resizeWindow("Frame", 800, 600)           # Resizes the window size for 'Frame' to 800x600    
cv.resizeWindow("Mask", 400, 300)            # Resizes the window size for 'Mask'  to 400x300

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------
    # Preprocesses 
    # -------------------------
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # -------------------------
    # Mask
    # -------------------------
    lower, upper = COLOR_RANGES[ASSIGNED_COLOR][0]
    mask = cv.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

    # -------------------------
    # Contours
    # -------------------------
    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    h, w, _ = frame.shape
    frame_cx = w // 2
    frame_cy = h // 2

    command = "SEARCH"

    if contours:
        gate = max(contours, key=cv.contourArea)
        area = cv.contourArea(gate)

        if area > 500:
            x, y, bw, bh = cv.boundingRect(gate)
            gate_cx = x + bw // 2
            gate_cy = y + bh // 2

            # Drawing gate and center
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.circle(frame, (gate_cx, gate_cy), 5, (0, 0, 255), -1)

            # -------------------------
            # 2D Alingment Logic
            # -------------------------
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
    cv.imshow("Mask", mask)

    #Unomment this to print the commands in the terminal
    #print(command)

    # Press Q to quit
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
