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

# 数値抽出（小数点付き対応）
def extract_number(text):
    match = re.search(r"(\d+\.\d+)", text.replace(",", ""))
    return float(match.group(1)) if match else None

# パラメータ抽出（改良版：ラベル行と数値行のペア処理）
def extract_parameters(img_pil):
    img_cv = pil_to_cv(img_pil)
    h, w = img_cv.shape[:2]

    # 画面左上をROIに設定
    roi = img_cv[int(h*0.1):int(h*0.5), int(w*0.02):int(w*0.3)]
    results = reader.readtext(roi)

    # 文字列と信頼度のみ抽出して前処理
    lines = [(text.strip(), conf) for _, text, conf in results if conf > 0.4]

    # キーワード定義（ゆるめ対応）
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
    i = 0
    while i < len(lines) - 1:
        label, _ = lines[i]
        value_line, _ = lines[i + 1]
        for key, variations in keywords.items():
            if any(kw.lower() in label.lower() for kw in variations):
                value = extract_number(value_line)
                if value is not None:
                    extracted[key] = value
        i += 1

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
