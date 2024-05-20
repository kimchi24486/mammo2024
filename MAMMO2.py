import streamlit as st
import cv2
import torch
from utils.hubconf import custom
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat
import io
# from ultralytics import YOLO
# Create an in-memory buffer for Excel
buffer = io.BytesIO()

p_time = 0
st.sidebar.markdown("<h2>TRƯỜNG: THCS MỸ HÒA</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>HSTH: PHẠM ĐỨC HÒA-NGUYỄN HỒNG NHÂN</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>HỘI THI “TIN HỌC TRẺ” </h3>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>CẤP THỊ LẦN THỨ VIII NĂM 2024</h3>", unsafe_allow_html=True)
st.sidebar.title('Cài đặt')
# Choose the model
model_type = st.sidebar.selectbox(
    'Chọn YOLOv7:', ('YOLO Model', 'YOLO', 'YOLOv7')
)

st.markdown("<h1 style='text-align: center; color: red;'>HỆ THỐNG PHÁT HIỆN VÀ PHÂN LOẠI VÙNG TỔN THƯƠNG TRÊN NHŨ ẢNH X-QUANG</h1>", unsafe_allow_html=True)
sample_img = cv2.imread('logo11.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')

if not model_type == 'YOLO Model':
    path_model_file = st.sidebar.text_input(
        f'Mô hình dự đoán: {model_type}',
        f'{model_type}.pt'
    )
    #if st.sidebar.checkbox('Tải mô hình'):
            # YOLOv7 Model
    if model_type == 'YOLOv7':
            # GPU
            gpu_option = st.sidebar.radio(
                'Chọn PU:', ('CPU', 'GPU'))

            if not torch.cuda.is_available():
                st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
            else:
                st.sidebar.success(
                    'GPU is Available on this Device, Choose GPU for the best performance',
                    icon="✅"
                )
            # Model
            if gpu_option == 'CPU':
                model = custom(path_or_model=path_model_file)
            if gpu_option == 'GPU':
                model = custom(path_or_model=path_model_file, gpu=True)

        # YOLOv8 Model

        # Load Class names
    class_labels = ["0: Mass","1: Global Asymmetry","2: Architectural Distortion","3: Nipple Retraction","4: Suspicious Calcification","5: Focal Asymmetry","6: Asymmetry","7: Skin Thickening","8: Suspicious Lymph Node","9: Skin Retraction"]
        # Inference Mode
        # Confidence
    confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
    draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=7
        )
        
    color_pick_list = []
    for i in range(len(class_labels)):
            classname = class_labels[i]
            color = color_picker_fn(classname, classname)
            color_pick_list.append(color)
  

        # Image

    upload_img_file = st.sidebar.file_uploader(
            'Tải hình ảnh lên:', type=['jpg', 'jpeg', 'png'])
    if upload_img_file is not None:
            pred = st.checkbox(f'Chọn dự đoán {model_type}')
            file_bytes = np.asarray(
                bytearray(upload_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            FRAME_WINDOW.image(img, channels='BGR')

            if pred:
                img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
                FRAME_WINDOW.image(img, channels='BGR')

               
                # Current number of classes
                class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                class_fq = json.dumps(class_fq, indent = 4)
                class_fq = json.loads(class_fq)
                df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        

                               
                # Updating Inference results
                with st.container():
                    st.markdown("<h2>Thống kê Suy luận</h2>", unsafe_allow_html=True)
                    st.markdown("<h3>Các đối tượng được phát hiện</h3>", unsafe_allow_html=True)
                    st.dataframe(df_fq, use_container_width=True)
    
            def convert_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            # Download buttons
            download_csv = st.download_button(
                label="Download data as CSV",
                data=convert_to_csv(df_fq),
                file_name='thongke_phanloaibenh.csv',
                mime='text/csv'
            )
                    

