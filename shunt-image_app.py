# ocr_streamlit_app.py

import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
import re

# EasyOCR Reader
reader = easyocr.Reader(['en'])  # 英語モデルのみ

def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def extract_number(text):
    match = re.search(r"(\d+\.\d+)", text.replace(",", ""))
    return float(match.group(1)) if match else None

def extract_parameters(img_pil):
    img_cv = pil_to_cv(img_pil)
    h, w = img_cv.shape[:2]

    # ROI (左上のラベル領域)
    roi = img_cv[int(h*0.1):int(h*0.5), int(w*0.02):int(w*0.3)]

    # OCR実行
    results = reader.readtext(roi)

    extracted = {}
    for _, text, conf in results:
        if conf < 0.4:
            continue
        text = text.strip()
        if "PS" in text:
            extracted["PSV"] = extract_number(text)
        elif "ED" in text:
            extracted["EDV"] = extract_number(text)
        elif "TAMAX" in text:
            extracted["TAMV"] = extract_number(text)
        elif "TAMEAN" in text or "TAV" in text:
            extracted["TAV"] = extract_number(text)
        elif "RI" in text:
            extracted["RI"] = extract_number(text)
        elif "PI" in text:
            extracted["PI"] = extract_number(text)
        elif "FV" in text:
            extracted["FV"] = extract_number(text)
        elif "VF Diam" in text:
            extracted["VF_Diam"] = extract_number(text)
    return extracted

# ---------------- Streamlit UI ----------------
st.title("EasyOCR: シャント機能数値抽出デモ")

uploaded = st.file_uploader("画像をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="入力画像", use_column_width=True)

    with st.spinner("OCR中..."):
        params = extract_parameters(img)

    if params:
        st.success("以下のパラメータが抽出されました：")
        st.json(params)
    else:
        st.warning("有効なパラメータが抽出できませんでした。ROIを調整してください。")
