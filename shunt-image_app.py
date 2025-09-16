# shunt_ocr_app.py
import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
import re

# OCRモデルの初期化（英語のみ）
reader = easyocr.Reader(['en'])

# PIL → OpenCV 変換
def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 数値抽出（小数点対応）
def extract_number(text):
    match = re.search(r"(\d+\.\d+)", text.replace(",", ""))
    return float(match.group(1)) if match else None

# パラメータ抽出（1行 or 2行に分かれたケース両対応）
def extract_parameters(img_pil):
    img_cv = pil_to_cv(img_pil)
    h, w = img_cv.shape[:2]

    # パラメータがある左上の範囲を抽出
    roi = img_cv[int(h * 0.1):int(h * 0.5), int(w * 0.02):int(w * 0.3)]
    results = reader.readtext(roi)

    # 読み取り結果を整形
    lines = [(text.strip(), conf) for _, text, conf in results if conf > 0.4]

    # 各パラメータのキーワード（曖昧対応）
    keywords = {
        "PSV": ["PS", "P5", "PSV"],
        "EDV": ["ED", "EDV"],
        "TAMV": ["TAMAX", "TA MAX"],
        "TAV": ["TAMEAN", "TAV"],
        "PI": ["PI"],
        "RI": ["RI"],
        "FV": ["FV"],
        "VF_Diam": ["VF Diam", "VF", "VFD"]
    }

    extracted = {}
    used_indices = set()

    # --- 1行内にラベル＋数値があるパターン ---
    for idx, (text, conf) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in text.lower() for kw in variations):
                value = extract_number(text)
                if value is not None:
                    extracted[key] = value
                    used_indices.add(idx)

    # --- 2行にラベルと数値が分かれているパターン ---
    i = 0
    while i < len(lines) - 1:
        if i in used_indices or i + 1 in used_indices:
            i += 1
            continue
        label, _ = lines[i]
        value_line, _ = lines[i + 1]
        for key, variations in keywords.items():
            if any(kw.lower() in label.lower() for kw in variations):
                value = extract_number(value_line)
                if value is not None:
                    extracted[key] = value
                    used_indices.update([i, i + 1])
        i += 1

    return extracted, results

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="シャントOCR", layout="centered")
st.title("🩺 シャント画像の数値自動抽出（EasyOCR）")

uploaded = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="入力画像", use_container_width=True)

    with st.spinner("OCR解析中..."):
        params, raw = extract_parameters(img)

    st.subheader("📊 抽出されたパラメータ")
    if params:
        st.json(params)
    else:
        st.warning("パラメータが見つかりませんでした")

    with st.expander("🔍 OCRの生データ（デバッグ用）"):
        for _, text, conf in raw:
            st.write(f"[{conf:.2f}] {text}")
