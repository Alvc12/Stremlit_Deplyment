import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR
import http.client
import json
import tempfile
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Driver Behavior Detection System")

model_path = "./.best (2).pt" 
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

if os.path.exists(model_path):
    yolo_model = YOLO(model_path)
else:
    st.error("Model file not found. Please check the path.")

confidence_threshold = st.sidebar.slider(
    'Confidence Threshold',
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.01
)
st.sidebar.title("Driver Behavior Detection System")
input_option = st.sidebar.selectbox("Select Detection Type", ("Image Processing", "Video Processing", "Camera Processing"))

def send_sms(custom_message):
    conn = http.client.HTTPSConnection("9klkx3.api.infobip.com")
    payload = json.dumps({
        "messages": [
            {
                "destinations": [{"to": "966508056428"}],
                "from": "447491163443",
                "text": custom_message
            }
        ]
    })
    headers = {
        'Authorization': 'App 86cde8061a25db1d5d0ec2b667c11951-0df99321-d263-444f-abb5-879f95519e9d',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    conn.request("POST", "/sms/2/text/advanced", payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    
    response_json = json.loads(data)
    if response_json.get("messages"):
        message_status = response_json["messages"][0]["status"]["name"]
        if message_status == "PENDING_ACCEPTED":
            return "Message sent successfully!"
        else:
            return f"Failed to send message: {message_status}"
    return "Failed to send message"

def model_detection():
    CAR_PLATTE = 0
    EATING_AND_DRINKING = 1
    USING_PHONE = 2
    
    
    if input_option == "Image Processing":
        st.subheader("Choose an image")
        image_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_file_uploader")
        if image_file is not None:
            img = Image.open(image_file)
            img_array = np.array(img)
            st.image(img, caption="Original Image", use_column_width=True)

            yolo_result = yolo_model(img_array)
            annotated_frame = yolo_result[0].plot() if len(yolo_result) > 0 else img_array
            st.image(annotated_frame, caption="Image after YOLO Application", use_column_width=True)

            detected_behaviors = []
            license_plate_text = ""

            for result in yolo_result[0].boxes.data.tolist():
                class_id = int(result[5])
                confidence = result[4]

                if class_id == CAR_PLATTE and confidence >= confidence_threshold:
                    xmin, ymin, xmax, ymax = map(int, result[:4])
                    license_plate_crop = img_array[ymin:ymax, xmin:xmax]
                    license_plate_text_result = ocr_model.ocr(license_plate_crop, cls=True)
                    license_plate_text = "\n".join([line[1][0] for line in license_plate_text_result[0]]) if license_plate_text_result else "No text found."
                    st.image(license_plate_crop, caption="Cropped License Plate", use_column_width=True)
                    st.subheader("Detected License Plate Text")
                    st.write(license_plate_text)

                if class_id in (EATING_AND_DRINKING, USING_PHONE):
                    detected_behaviors.append("Dear driver, please pay attention to your driving.")
            
            if detected_behaviors:
                custom_message = " | ".join(detected_behaviors)
                with st.spinner("Sending message..."):
                    response = send_sms(custom_message)
                st.success(response)

    elif input_option == "Video Processing":
        st.subheader("Choose a video")
        video_file = st.file_uploader("", type=["mp4", "avi", "mov"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            vid = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            text_detected = False
            behaviors_detected = False

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                yolo_result = yolo_model(frame)
                annotated_frame = yolo_result[0].plot() if len(yolo_result) > 0 else frame

                detected_behaviors = []
                license_plate_text = ""

                for result in yolo_result[0].boxes.data.tolist():
                    class_id = int(result[5])
                    confidence = result[4]
                    
                    if class_id == CAR_PLATTE and confidence >= confidence_threshold:
                        xmin, ymin, xmax, ymax = map(int, result[:4])
                        license_plate_crop = frame[ymin:ymax, xmin:xmax]
                        license_plate_text_result = ocr_model.ocr(license_plate_crop, cls=True)
                        license_plate_text = "\n".join([line[1][0] for line in license_plate_text_result[0]]) if license_plate_text_result else "No text found."
                        st.write(f"Detected License Plate Text: {license_plate_text}")

                    if confidence >= confidence_threshold and class_id in (EATING_AND_DRINKING, USING_PHONE):
                        detected_behaviors.append("Dear driver, please pay attention to your driving.")
                
                if detected_behaviors and not behaviors_detected:
                    custom_message = " | ".join(detected_behaviors)
                    with st.spinner("Sending message..."):
                        response = send_sms(custom_message)
                    st.success(response)
                    behaviors_detected = True
                
                stframe.image(annotated_frame, channels="BGR")

            vid.release()

    elif input_option == "Camera Processing":
        st.subheader("Camera Processing")
        
        if 'camera_open' not in st.session_state:
            st.session_state.camera_open = False
        
        if st.button("Open/Close Camera", key="toggle_camera"):
            st.session_state.camera_open = not st.session_state.camera_open
            
            if st.session_state.camera_open:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                st.write("Camera is open. Click 'Open/Close Camera' to close.")

                behaviors_detected = False

                while st.session_state.camera_open:
                    ret, frame = cap.read()
                    if not ret:
                        st.write("Failed to capture frame. Please check your camera connection.")
                        break

                    yolo_result = yolo_model(frame)
                    annotated_frame = yolo_result[0].plot() if len(yolo_result) > 0 else frame

                    detected_behaviors = []
                    license_plate_text = ""

                    for result in yolo_result[0].boxes.data.tolist():
                        class_id = int(result[5])
                        confidence = result[4]
                        
                        if class_id == CAR_PLATTE and confidence >= confidence_threshold:
                            xmin, ymin, xmax, ymax = map(int, result[:4])
                            license_plate_crop = frame[ymin:ymax, xmin:xmax]
                            license_plate_text_result = ocr_model.ocr(license_plate_crop, cls=True)
                            license_plate_text = "\n".join([line[1][0] for line in license_plate_text_result[0]]) if license_plate_text_result else "No text found."
                            st.write(f"Detected License Plate Text: {license_plate_text}")

                        if confidence >= confidence_threshold and class_id in (EATING_AND_DRINKING, USING_PHONE):
                            detected_behaviors.append("Dear driver, please pay attention to your driving.")
                    
                    if detected_behaviors and not behaviors_detected:
                        custom_message = " | ".join(detected_behaviors)
                        with st.spinner("Sending message..."):
                            response = send_sms(custom_message)
                        st.success(response)
                        behaviors_detected = True

                    stframe.image(annotated_frame, channels="BGR")

                cap.release()
                st.write("Camera closed.")

model_detection()



st.sidebar.subheader("Driver Statistics")

st.sidebar.subheader("Using a phone during driving")
labels1 = ['Using', 'Not Using']
sizes1 = [53.6, 46.4]
colors1 = ['#ff9999', '#66b3ff']
explode1 = (0.1, 0)

fig1, ax1 = plt.subplots()
fig1.patch.set_facecolor('none')  
ax1.pie(sizes1, explode=explode1, labels=labels1, colors=colors1,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  
st.sidebar.pyplot(fig1)

st.sidebar.subheader("Drivers Who Had Accidents While Using the Phone")
labels2 = ['Had', 'Hadn\'t']
sizes2 = [32.1, 67.9]
colors2 = ['#ffcc99', '#99ff99']
explode2 = (0.1, 0)

fig2, ax2 = plt.subplots()
fig2.patch.set_facecolor('none')  
ax2.pie(sizes2, explode=explode2, labels=labels2, colors=colors2,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax2.axis('equal')  
st.sidebar.pyplot(fig2)

