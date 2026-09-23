import cv2
import numpy as np
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading YuNet face detection model ({MODEL_FILENAME}) ...")
        r = requests.get(url)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"Saved model to {path}")
        else:
            raise RuntimeError(f"Failed to download model (status {r.status_code})")
    else:
        print(f"Found existing model: {path}")

download_file(MODEL_URL, MODEL_PATH)

print("Loading YuNet face detector ...")
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    None,
    (320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

image_path = os.path.join(BASE_DIR, "test_preview.jpg")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Input image not found: {image_path}")

print(f"Reading image: {image_path}")
try:
    from PIL import Image
    pil_img = Image.open(image_path)
    pil_img = pil_img.convert("RGB")
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
except Exception as e:
    raise RuntimeError(f"Failed to read image with PIL: {e}")

if image is None:
    raise ValueError(f"Failed to read image from {image_path}")

(h, w) = image.shape[:2]
detector.setInputSize((w, h))

print("Running face detection inference ...")
retval, faces = detector.detect(image)

if faces is not None:
    print(f"Detected {len(faces)} face(s)")
    for i, face in enumerate(faces):
        x, y, face_w, face_h = map(int, face[:4])
        cv2.rectangle(image, (x, y), (x + face_w, y + face_h), (0, 255, 0), 2)
        cv2.putText(image, f"Face {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
    print("No faces detected.")

output_path = os.path.join(BASE_DIR, "output_with_faces.jpg")
cv2.imwrite(output_path, image)
print(f"Saved annotated image as {output_path}")
print("Face detection complete.")
