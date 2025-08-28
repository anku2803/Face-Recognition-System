import cv2
import os
import sqlite3

def insert_or_update(id, name):
    conn = sqlite3.connect('FaceBase.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS people (person_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    cursor.execute("SELECT * FROM people WHERE person_id=?", (id,))
    record = cursor.fetchone()

    if record:
        cursor.execute("UPDATE people SET name=? WHERE person_id=?", (name, id))
    else:
        cursor.execute("INSERT INTO people (person_id, name) VALUES (?, ?)", (id, name))
    
    conn.commit()
    conn.close()

# Input ID and Name
id = int(input("Enter ID: "))
name = input("Enter Name: ")

insert_or_update(id, name)

# Create folder
folder = f'dataset/{id}'
os.makedirs(folder, exist_ok=True)

# Start camera and capture 20 images
cap = cv2.VideoCapture(0)
count = 0

while count < 20:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    img_path = os.path.join(folder, f'{name}_{count}.jpg')
    cv2.imwrite(img_path, frame)
    cv2.imshow('Saving Face Data', frame)
    cv2.waitKey(100)

print("✅ Data collection completed")
cap.release()
cv2.destroyAllWindows()
