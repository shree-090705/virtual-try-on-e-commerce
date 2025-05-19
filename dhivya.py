import cv2
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the sunglasses image (with white background)
raw_sunglass = cv2.imread("glass.jpg")  # Use PNG or JPEG with white background
if raw_sunglass is None:
    raise FileNotFoundError("glass.jpg not found.")

# Convert white background to transparent
gray = cv2.cvtColor(raw_sunglass, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
b, g, r = cv2.split(raw_sunglass)
alpha = mask
sunglass_img = cv2.merge([b, g, r, alpha])

# Function to overlay image with alpha channel
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background
    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0

    roi = background[y:y+h, x:x+w]
    blended = roi * (1 - mask) + overlay_img * mask
    background[y:y+h, x:x+w] = blended.astype(np.uint8)
    return background

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

    for (x, y, w, h) in faces:
        # Estimate position and size for sunglasses
        sunglass_width = w
        scale = sunglass_width / sunglass_img.shape[1]
        sunglass_height = int(sunglass_img.shape[0] * scale)
        resized_sunglass = cv2.resize(sunglass_img, (sunglass_width, sunglass_height), interpolation=cv2.INTER_AREA)

        # Positioning over the eyes (about 35% down from top of face)
        sunglass_x = x
        sunglass_y = y + int(h * 0.35) - sunglass_height // 2

        # Overlay sunglasses
        frame = overlay_transparent(frame, resized_sunglass, sunglass_x, sunglass_y)
        break  # only process the first detected face

    cv2.imshow("Virtual Try-On Sunglasses (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()