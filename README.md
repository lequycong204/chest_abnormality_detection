# Chest X-ray Abnormality Detection App

Ứng dụng **Streamlit** dùng để **phát hiện bất thường trên ảnh X-ray ngực** (Chest X-ray), hỗ trợ inference bằng **YOLOv5** và **RetinaNet**, có khả năng đọc **ảnh thường (PNG/JPG)** và **DICOM**.

---

## 📌 Tính năng chính

- 📷 Upload ảnh X-ray định dạng: `PNG`, `JPG`, `JPEG`, `DICOM (.dcm)`
- 🤖 Hỗ trợ **mô hình detection**:
  - **YOLOv5**
  - **RetinaNet (ResNet50 + FPN)**
  - **Faster R-CNN**
- 🩺 Phát hiện nhiều loại bất thường ngực
- 📊 Hiển thị bounding box + nhãn + confidence score
- ⚡ Giao diện trực quan bằng **Streamlit**

---

## 🧠 Các lớp bệnh được hỗ trợ

```text
Aortic enlargement
Atelectasis
Calcification
Cardiomegaly
Consolidation
ILD
Infiltration
Lung Opacity
Nodule/Mass
Other lesion
Pleural effusion
Pleural thickening
Pneumothorax
Pulmonary fibrosis
```

---
## 🏗️ Kiến trúc tổng quan
```
├── app.py                 # Streamlit
├── weights/
│   ├── yolo/
│   │   └── best.pt
│   └── retina/
│       └── weight_retina_0.195.pth
├── README.md
```

## ⚙️ Yêu cầu môi trường
- Python >= 3.8
- PyTorch
- torchvision
- torchmetrics
- OpenCV
- Streamlit
- pydicom
- numpy
- bbox_visualizer

## Cài đặt dependencies
`pip install -r requirements.txt`

## 🚀 Cách chạy ứng dụng
`streamlit run app.py`

Sau đó mở trình duyệt tại:

`http://localhost:8501`

