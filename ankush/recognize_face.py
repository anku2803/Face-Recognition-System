import cv2
import torch
import joblib
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3

# Load trained models
knn_model = joblib.load('knn_face_recognizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)

# Load FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# SQLite function to fetch profile
def get_profile(person_id):
    try:
        conn = sqlite3.connect("FaceBase.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM people WHERE person_id=?", (int(person_id),))
        profile = cursor.fetchone()
        conn.close()
        return profile
    except sqlite3.Error as e:
        print("Database error:", e)
        return None

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

print("🟢 Face recognition started (Press ESC to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    boxes, probs = mtcnn.detect(img_pil)

    if boxes is not None:
        faces = mtcnn(img_pil)
        
        for i, box in enumerate(boxes):
            if box is None or faces is None or len(faces) <= i:
                continue

            x1, y1, x2, y2 = map(int, box)
            face_tensor = faces[i]

            if face_tensor is None:
                continue

            # Ensure shape [1, 3, 160, 160]
            if face_tensor.ndim == 3:
                face_tensor = face_tensor.unsqueeze(0)

            face_tensor = face_tensor.to(device)

            try:
                with torch.no_grad():
                    embedding = facenet(face_tensor).cpu().numpy()

                pred_id = knn_model.predict(embedding)[0]
                real_id = label_encoder.inverse_transform([pred_id])[0]
                dist, _ = knn_model.kneighbors(embedding, n_neighbors=1)
                confidence = dist[0][0]

                print(f"→ Predicted ID: {real_id}, Distance: {confidence:.2f}")

                if confidence < 0.7:
                    profile = get_profile(real_id)
                    if profile:
                        label = f"{profile[1]} (ID: {profile[0]})"
                        color = (0, 255, 0)  # Green
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)  # Red
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                print(f"Recognition error: {e}")

    # Display result
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
