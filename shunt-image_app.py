# shunt_ocr_app.py
import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
import re

# OCRモデル初期化
reader = easyocr.Reader(['en'])

# PIL → OpenCV変換
def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 数値抽出
def extract_number(text):
    match = re.search(r"(\d+\.\d+)", text.replace(",", ""))
    return float(match.group(1)) if match else None

# パラメータ抽出（ゆるめのマッチング対応）
def extract_parameters(img_pil):
    img_cv = pil_to_cv(img_pil)
    h, w = img_cv.shape[:2]

    # 画像の左上（ラベル表示部）を切り出し
    roi = img_cv[int(h*0.1):int(h*0.5), int(w*0.02):int(w*0.3)]

    results = reader.readtext(roi)

    # 各パラメータに対応する複数表現
    keywords = {
        "PSV": ["PS", "P5", "P5V", "PSV"],
        "EDV": ["ED", "EDV", "EQ"],
        "TAMV": ["TAMAX", "TA MAX"],
        "TAV": ["TAMEAN", "TA MEAN", "TAV"],
        "PI": ["PI"],
        "RI": ["RI"],
        "FV": ["FV"],
        "VF_Diam": ["VF Diam", "VF", "VFD"]
    }

    extracted = {}
    for _, text, conf in results:
        if conf < 0.4:
            continue  # 信頼度が低すぎるものはスキップ
        for key, variations in keywords.items():
            if any(kw.lower() in text.lower() for kw in variations):
                val = extract_number(text)
                if val is not None:
                    extracted[key] = val
    return extracted, results

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="シャント画像OCR", layout="centered")
st.title("🩸 シャント画像から数値を自動抽出（EasyOCR）")

uploaded = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="入力画像", use_container_width=True)

    with st.spinner("OCRで解析中..."):
        params, ocr_raw = extract_parameters(img)

    st.subheader("📊 抽出されたパラメータ")
    if params:
        st.json(params)
    else:
        st.warning("パラメータが検出できませんでした。画像の解像度や切り出し範囲をご確認ください。")

    # デバッグ用のOCRテキスト表示
    with st.expander("🔍 OCRの生出力（デバッグ用）"):
        for _, text, conf in ocr_raw:
            st.write(f"[{conf:.2f}] {text}")
