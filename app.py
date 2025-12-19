import streamlit as st
import torch
import cv2
import numpy as np
import pydicom
import warnings
warnings.filterwarnings("ignore")

from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

# 1. LOAD YOLOv5
def load_yolov5(weight_path):
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weight_path, force_reload=False)
    return model

# 2. LOAD RETINANET
def load_retinanet(weight_path, num_classes=15):
    model = retinanet_resnet50_fpn_v2(weights=None)

    # Replace classification head
    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes
    )

    ckpt = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

# 3. XỬ LÝ DICOM
def read_dicom(file):
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array.astype(np.float32)

    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# 4. VẼ BOUNDING BOX + CLASS NAME
def draw_boxes(image, boxes, labels, scores=None, class_names=None):
    img = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # Vẽ box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Lấy tên class
        label_text = ""
        if labels is not None:
            if class_names is not None:
                label_text = class_names[int(labels[i])]
            else:
                label_text = str(labels[i])

        # Thêm score
        if scores is not None:
            label_text += f" {scores[i]:.2f}"

        # Vẽ label
        cv2.putText(img, label_text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    return img

# 5. YOLOv5 INFERENCE
def predict_yolo(model, image):
    results = model(image)
    df = results.pandas().xyxy[0]

    if len(df) == 0:
        return None, None, None

    boxes = df[["xmin", "ymin", "xmax", "ymax"]].values
    labels = df["class"].values
    scores = df["confidence"].values

    return boxes, labels, scores


# 6. RETINANET INFERENCE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_retinanet(image):
    """
    image: numpy (H, W) GRAYSCALE hoặc (H, W, 3)
    """

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    # float32 để tránh lỗi albumentations behavior
    image = image.astype(np.float32)

    # Scale giống max_pixel_value=255.0 của Albumentations
    image = image / 255.0

    # Normalize theo ImageNet (CHÍNH XÁC như training)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW
    tensor = torch.from_numpy(image).permute(2, 0, 1)

    return tensor


def predict_retinanet(model, image, device="cuda" if torch.cuda.is_available() else "cpu", score_thresh=0.3):
    """
    image: numpy (H,W) hoặc (H,W,3), kiểu uint8
    """
    model.eval()
    model.to(device)

    img_tensor = preprocess_retinanet(image) 
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    scores = outputs["scores"].cpu()
    keep = scores > score_thresh

    if keep.sum() == 0:
        return None, None, None

    boxes  = outputs["boxes"][keep].cpu().numpy()
    labels = outputs["labels"][keep].cpu().numpy()
    scores = outputs["scores"][keep].cpu().numpy()

    return boxes, labels, scores

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# ================================
# 7. STREAMLIT UI (SIDEBAR + LOADING)
# ================================

st.title("🔎 Ứng dụng phát hiện bất thường trên X-ray ngực")

# ==== SIDEBAR CONFIG ==== 
st.sidebar.header("⚙️ Cấu hình")

path_yolo = "D:\\AdCV_VinChestXray\\App\\weights\\yolo\\best.pt"
path_retina = "D:\\AdCV_VinChestXray\\App\\weights\\retina\\weight_retina_0.195.pth"

model_type = st.sidebar.selectbox("Chọn model:", ["YOLOv5", "RetinaNet"])

uploaded_file = st.sidebar.file_uploader(
    "Upload PNG/JPG/DICOM", 
    type=["png", "jpg", "jpeg", "dcm"]
)

run_button = st.sidebar.button("🚀 Chạy Inference")

st.sidebar.markdown("---")
st.sidebar.info("📌 Upload ảnh và nhấn **Chạy Inference**")

# ==== MAIN PANEL ====
st.header("📌 Kết quả hiển thị")

if uploaded_file is None:
    st.info("⬅ Hãy upload ảnh trong SIDEBAR")
else:
    # Load ảnh
    if uploaded_file.type == "application/dicom":
        img = read_dicom(uploaded_file)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2 ảnh lớn hơn nhờ sidebar
    col1, col2 = st.columns(2)

    # Hiển thị ảnh gốc
    with col1:
        st.subheader("📷 Ảnh gốc")
        st.image(img, width=600)

    if run_button:
        with st.spinner("⏳ Đang chạy mô hình… vui lòng chờ..."):
            if model_type == "YOLOv5":
                model = load_yolov5(path_yolo)
                boxes, labels, scores = predict_yolo(model, img)
            else:
                model = load_retinanet(path_retina)
                boxes, labels, scores = predict_retinanet(model, img)

        if boxes is None or len(boxes) == 0:
            st.warning("✔ Không phát hiện bất thường nào!")
        else:
            pred_img = draw_boxes(
                img,
                boxes,
                labels,
                scores=scores,
                class_names=CLASS_NAMES
            )

            # Hiển thị ảnh kết quả
            with col2:
                st.subheader("📊 Ảnh kết quả")
                st.image(pred_img, width=600)

            st.success(f"🔍 Phát hiện **{len(boxes)}** vùng bất thường")
