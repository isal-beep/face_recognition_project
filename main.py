import cv2
import os
import numpy as np
import requests
from datetime import datetime

# === PATH ===
DATASET_PATH = 'dataset'
MODEL_PATH = 'models'

AGE_PROTO = os.path.join(MODEL_PATH, 'age_deploy.prototxt')
AGE_MODEL = os.path.join(MODEL_PATH, 'age_net.caffemodel')
GENDER_PROTO = os.path.join(MODEL_PATH, 'gender_deploy.prototxt')
GENDER_MODEL = os.path.join(MODEL_PATH, 'gender_net.caffemodel')

# === MODEL UMUR & GENDER ===
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['male', 'female']

# === MODEL FACE RECOGNITION ===
recognizer = cv2.face.LBPHFaceRecognizer_create()

# === URL API BACKEND (pastikan sesuai) ===
API_URL = "http://localhost:3000/api/attendance"

# === SIMPAN ABSEN KE SERVER ===
def simpan_ke_server(nama):
    now = datetime.now()
    tanggal = now.strftime("%Y-%m-%d")
    waktu = now.strftime("%H:%M:%S")

    data = {
        "name": nama,
        "tanggal": tanggal,
        "waktu": waktu,
        "status": "Hadir"
    }

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code in [200, 201]:
            print(f"[✅ SUKSES] {nama} absen pada {tanggal} {waktu}")
            print("Respon server:", response.text)
        else:
            print(f"[❌ ERROR] Gagal kirim data ke server: {response.text}")
    except Exception as e:
        print(f"[⚠] Tidak bisa terhubung ke server: {e}")

# === LOAD DATASET WAJAH ===
def get_images_and_labels(dataset_path):
    faces = []
    labels = []
    label_names = {}

    # Daftar orang yang dikenal
    person_list = ['faisal', 'sayid']

    for label, person_name in enumerate(person_list):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_names[label] = person_name

        for filename in os.listdir(person_folder):
            if filename.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(person_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    faces.append(image)
                    labels.append(label)

    return faces, np.array(labels), label_names

# === MULAI PROGRAM ===
faces, labels, label_names = get_images_and_labels(DATASET_PATH)
recognizer.train(faces, labels)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
print("Tekan 'q' untuk keluar")

sudah_absen_nama = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227),
                                     (78.426, 87.77, 114.895), swapRB=False)

        # Prediksi umur & gender
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        try:
            label, confidence = recognizer.predict(roi_gray)
            nama = label_names.get(label, "unknown")
            color = (0, 255, 0) if confidence < 100 else (0, 0, 255)

            if confidence < 100:
                display_text = f"{nama} | {gender} | {age}"
                if nama not in sudah_absen_nama:
                    simpan_ke_server(nama)
                    sudah_absen_nama.add(nama)
            else:
                display_text = "Unknown"
        except Exception as e:
            display_text = "Unknown"
            color = (0, 0, 255)
            print("[ERROR] Gagal mengenali wajah:", e)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, display_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
