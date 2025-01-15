from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

app = FastAPI()

# โหลด Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.post("/detect/")
async def detect_faces(file: UploadFile = File(...)):
    # อ่านไฟล์ภาพ
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # ตรวจจับใบหน้า
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # ส่งตำแหน่งใบหน้ากลับ
    results = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
    return {"faces": results}
