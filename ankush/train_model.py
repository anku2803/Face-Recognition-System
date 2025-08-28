import os
import torch
import numpy as np
import joblib
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

data_dir = 'dataset'
embeddings, labels = [], []

for person_id in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_id)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                emb = facenet(face.unsqueeze(0).to(device)).cpu().numpy()
            embeddings.append(emb[0])
            labels.append(int(person_id))  # Use ID from folder name

embeddings = np.array(embeddings)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(embeddings, encoded_labels)

joblib.dump(knn, 'knn_face_recognizer.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("✅ Model training complete.")
