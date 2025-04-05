from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2
import pandas as pd

# โหลดโมเดลและข้อมูลที่เกี่ยวข้อง
model = load_model('model.h5')
label_map = np.load('label.npy', allow_pickle=True).item()
csv_file = 'Gemstones_info.csv'
info = pd.read_csv(csv_file)
Gemstones_info_th = info.set_index('Name').to_dict()['Name_th']

# ฟังก์ชันทำนายภาพ
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256)) #ปรับขนาด
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array) #ส่งภาพที่เตรียมไปยังโมเดล
    predicted_label = np.argmax(predictions[0]) # หาค่าประเภทที่มีโอกาสมากที่สุดด้วย
    confidence = predictions[0][predicted_label] #คํานวนค่าความใกล้
    predicted_name_en = label_map.get(predicted_label, 'Unknown') #เเปลงเป็นชื่ออังกฤษ
    predicted_name_th = Gemstones_info_th.get(predicted_name_en, 'Unknown') #ชื่อไทย
    return predicted_name_en, predicted_name_th, confidence * 100 # ฟังก์ชันส่งคืนค่า 3 อย่าง

# ฟังก์ชันตรวจจับวัตถุหลายชนิดในภาพเดียว
def detect_multiple_objects_with_labels(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #เเปลงภาพเป็นสีเทา เพื่อประมวลผลง่าย
    image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(image_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2) #เเปลงภาพเป็นขาวดํา
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #ค้นหาขอบวัตถุเพื่อระบุรุปร่าง

    min_area = 1000
    detected_objects = []

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = image[y:y + h, x:x + w] #ใช้cv2.contourAreaเพื่อหากรอบ

            # Save temporary image for prediction
            temp_image_path = f"temp_object.jpg" #บันทึกวัตถุที่ครอบเเละเรียกใช้ฟัีงก์ชั่น
            cv2.imwrite(temp_image_path, cropped_image)  #เรียกใช้ฟังก์ชันเพื่อให้เเสดงผล

            # Predict using predict_image
            name_en, name_th, confidence = predict_image(temp_image_path) #เก็บข้อมูลที่ตรวจพบลง predict image
            detected_objects.append((name_en, name_th, confidence, (x, y, w, h)))

            # Remove the temporary file
            os.remove(temp_image_path) #ลบไฟล์ภาพชั่วคราวที่ใช้ในขั้นตอนการพยากรณ์ เพื่อลดการใช้พื้นที่เก็บข้อมูล

    return image, detected_objects

# สร้างแอป Flask
app = Flask(__name__)

# Route สำหรับหน้าแรก
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route สำหรับการทำนาย
@app.route("/predict/", methods=["POST"])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template("index.html", error="No file selected. Please try again.")
# ตรวจสอบว่ามีการอัพโหลดไฟล์ไหม


    file = request.files['file']
    try:
        # บันทึกไฟล์อัปโหลดชั่วคราว
        temp_image_path = "uploaded_image.jpg"
        file.save(temp_image_path)

        # เรียกใช้ฟังกืชั่น detect_multiple_objects เพื่อตรวจจับวัตถุหลายชนิด
        processed_image, detected_objects = detect_multiple_objects_with_labels(temp_image_path)

        # วาดกรอบและแสดงข้อมูล
        for _, _, _, (x, y, w, h) in detected_objects:
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # บันทึกผลลัพธ์ในโฟลเดอร์ static เพื่อให้เเสดงผลหน้าเว็บได้
        result_image_path = "static/result.jpg"
        cv2.imwrite(result_image_path, processed_image)

        # ลบไฟล์ชั่วคราว เพื่อลดพื้นที่ในการเก้บข้อมูล
        os.remove(temp_image_path)

        # ส่งผลลัพธ์ให้เเสดงไปที่หน้าเว็บไปยังหน้าเว็บ
        return render_template(
            "result.html", 
            objects=[{
                'name_en': obj[0], 
                'name_th': obj[1], 
                'confidence': f"{float(obj[2]):.2f}%" if isinstance(obj[2], (float, int)) else "N/A", 
                'box': obj[3]
            } for obj in detected_objects],
            result_image=result_image_path
        )

    except Exception as e:
        return render_template("index.html", error=str(e))
    #ตรวจข้อผิดพลาด

# รันแอป
if __name__ == "__main__":
    app.run(debug=True)
