# easyocr_app.py
# 画像からシャント機能評価パラメータをEasyOCRで抽出

import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re

# EasyOCR reader 初期化（英語）
reader = easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="EasyOCR パラメータ抽出", layout="wide")
st.title("透析シャント画像からパラメータ抽出（EasyOCR版）")

uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="アップロード画像", use_container_width=True)

    # NumPy配列に変換
    img = np.array(pil_img)
    h, w, _ = img.shape

    # 左上のラベルボックス相当のROIを切り出す（おおよその範囲）
    x0, x1 = int(w * 0.02), int(w * 0.22)
    y0, y1 = int(h * 0.18), int(h * 0.42)
    roi = img[y0:y1, x0:x1]

    st.subheader("ROI: ラベル領域（左上）")
    st.image(roi, caption="推定ラベル領域", channels="RGB")

    with st.spinner("OCR実行中..."):
        results = reader.readtext(roi, detail=1)

    # 表示用に文字と信頼度だけ抽出
    raw_texts = [(text, conf) for (_, text, conf) in results if conf > 0.3]

    st.subheader("OCR生データ")
    for text, conf in raw_texts:
        st.write(f"[{conf:.2f}] {text}")

    # 数値抽出ロジック
    PARAM_KEYS = ["PS", "ED", "TAMAX", "TAMEAN", "PI", "RI", "FV", "VF Diam"]
    param_dict = {}

    for text, conf in raw_texts:
        for key in PARAM_KEYS:
            if key.lower() in text.lower():
                # 数値部分を抽出
                match = re.search(r"(-?\d+(?:[.,]\d+)?)", text)
                if match:
                    val = float(match.group(1).replace(",", "."))
                    param_dict[key] = val

    st.subheader("抽出結果")
    if param_dict:
        st.json(param_dict)
    else:
        st.warning("有効なパラメータが検出できませんでした。")
