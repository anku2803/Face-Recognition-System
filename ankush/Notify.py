import cv2
import torch
import joblib
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
from plyer import notification
import winsound
import time
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os
from datetime import datetime

# ============================
# ✅ Load environment variables
# ============================
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# ============================
# ✅ Load trained models
# ============================
knn_model = joblib.load('knn_face_recognizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ============================
# ✅ Fetch profile from SQLite
# ============================
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

# ============================
# ✅ Show desktop notification
# ============================
def send_notification(title, message):
    notification.notify(title=title, message=message, timeout=5)
    winsound.Beep(1000, 500)

# ============================
# ✅ Send Email with snapshot
# ============================
def send_email_alert(snapshot_path):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    msg = EmailMessage()
    msg['Subject'] = f'🔴 ALERT: Unknown Person Detected @ {timestamp}'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS  # Send to self

    msg.set_content(f'⚠️ An unknown person was detected at {timestamp}.\nSnapshot attached for your review.')

    # Attach snapshot
    with open(snapshot_path, 'rb') as f:
        img_data = f.read()
    msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='intruder.jpg')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print(f"✅ Email alert sent at {timestamp}.")

# ============================
# ✅ Unknown detection control
# ============================
unknown_start_time = None
unknown_alert_sent = False
UNKNOWN_DURATION_THRESHOLD = 5  # seconds

# ============================
# ✅ Start webcam
# ============================
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

    unknown_detected = False

    if boxes is not None:
        faces = mtcnn(img_pil)

        for i, box in enumerate(boxes):
            if box is None or faces is None or len(faces) <= i:
                continue

            x1, y1, x2, y2 = map(int, box)
            face_tensor = faces[i]

            if face_tensor is None:
                continue

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
                        color = (0, 255, 0)
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)
                        unknown_detected = True
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                    unknown_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                print(f"Recognition error: {e}")

    # ============================
    # ✅ Handle unknown detection
    # ============================
    if unknown_detected:
        send_notification("Unknown Face Detected", "An unrecognized person was detected!")
        if unknown_start_time is None:
            unknown_start_time = time.time()
        elif (time.time() - unknown_start_time) >= UNKNOWN_DURATION_THRESHOLD:
            if not unknown_alert_sent:
                snapshot_path = "intruder.jpg"
                cv2.imwrite(snapshot_path, frame)
                send_email_alert(snapshot_path)
                unknown_alert_sent = True
    else:
        unknown_start_time = None
        unknown_alert_sent = False

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
