import cv2
import torch
from PIL import Image

from config import DEVICE, MODEL_PATH, FACE_DETECTION_MODEL
from model import create_model
from transforms import test_transforms

face_detector = cv2.CascadeClassifier(
    FACE_DETECTION_MODEL
)

CLASS_NAMES = [
    "with_mask",
    "without_mask"
]

model = create_model().to(DEVICE)
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)
model.eval()

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face = Image.fromarray(face)

        face = test_transforms(face)

        face = face.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = model(face)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        label = CLASS_NAMES[prediction.item()]
        confidence = confidence.item()

        if label == "with_mask":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{label} {confidence*100:.1f}%",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "Face Mask Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()