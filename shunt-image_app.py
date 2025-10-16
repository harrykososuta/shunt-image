
import streamlit as st
import pytesseract
import numpy as np
from PIL import Image
import cv2
import re
from collections import OrderedDict

# =============================
# OCR処理関連（pytesseract使用）
# =============================

def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def preprocess_image(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text_from_image(img_cv):
    preprocessed = preprocess_image(img_cv)
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(preprocessed, config=custom_config)

def extract_parameters_from_text(text):
    pattern_dict = {
        "PSV": r"(?:PSV|PS)[^\d]*(\d+\.\d+)",
        "EDV": r"(?:EDV|ED)[^\d]*(\d+\.\d+)",
        "TAMV": r"(?:TAMAX|TAMV|TA MAX)[^\d]*(\d+\.\d+)",
        "TAV": r"(?:TAMEAN|TAV|TA MEAN)[^\d]*(\d+\.\d+)",
        "PI": r"PI[^\d]*(\d+\.\d+)",
        "RI": r"RI[^\d]*(\d+\.\d+)",
        "FV": r"FV[^\d]*(\d+\.\d+)",
        "VF_Diam": r"VF\s*Diam[^\d]*(\d+\.\d+)"
    }

    extracted = OrderedDict()
    for key, pattern in pattern_dict.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted[key] = float(match.group(1))
    return extracted

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="シャントOCR", layout="centered")
st.title("🩺 シャント画像の数値自動抽出＆診断（pytesseract）")

uploaded = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="入力画像", use_container_width=True)

    with st.spinner("OCR解析中..."):
        img_cv = pil_to_cv(img)
        ocr_text = extract_text_from_image(img_cv)
        params = extract_parameters_from_text(ocr_text)

    st.subheader("📊 抽出されたパラメータ")
    if params:
        st.json(params)
    else:
        st.warning("パラメータが見つかりませんでした")

    st.subheader("📝 OCR全文")
    st.text_area("OCR結果全文", ocr_text, height=300)


    # ===== 評価スコア =====
    st.subheader("🔍 自動評価スコア")
    form = {k.lower(): v for k, v in params.items()}
    score = 0
    comments = []

    if form.get("tav", 999) <= 34.5:
        score += 1
        comments.append(("warning", "TAVが34.5 cm/s以下 → 低血流が疑われる"))
    if form.get("ri", 0) >= 0.68:
        score += 1
        comments.append(("warning", "RIが0.68以上 → 高抵抗が疑われる"))
    if form.get("pi", 0) >= 1.3:
        score += 1
        comments.append(("warning", "PIが1.3以上 → 脈波指数が高い"))
    if form.get("edv", 999) <= 40.4:
        score += 1
        comments.append(("warning", "EDVが40.4 cm/s以下 → 拡張期血流速度が低い"))

    st.write(f"評価スコア: {score} / 4")
    if score == 0:
        st.success("🟢 正常：経過観察が推奨されます")
    elif score in [1, 2]:
        st.warning("🟡 要注意：追加評価が必要です")
    else:
        st.error("🔴 高リスク：専門的評価が必要です")

    if comments:
        st.write("### 評価コメント")
        for level, comment in comments:
            st.warning(f"- {comment}")

    # ===== 波形分類 =====
    st.subheader("📈 波形分類結果")

    def classify_waveform(psv, edv, pi, fv):
        if edv < 5 and fv < 100:
            return "Type V", "閉塞型（Ⅴ型）：EDVほぼゼロ・流量非常に低い"
        elif fv > 1500:
            return "Type I", "過大血流型（Ⅰ型）：FV が 1500 を超える"
        elif pi >= 1.3 and edv < 40.4:
            return "Type IV", "末梢狭窄型（Ⅳ型）：PI 高値、EDV やや低下"
        elif pi >= 1.3:
            return "Type III", "狭窄傾向（Ⅲ型）：PI 高値"
        elif fv < 500 and edv < 40.4:
            return "Type IV", "末梢狭窄型（Ⅳ型）：FV 低値 & EDV やや低下"
        else:
            return "Type II", "良好波形型（Ⅱ型）：EDV 保たれ、PI 正常域"

    with st.expander("📊 波形分類と説明（クリックで展開）"):
        psv = params.get("PSV", 0)
        edv = params.get("EDV", 0)
        pi = params.get("PI", 0)
        fv = params.get("FV", 0)
        if all([psv, edv, pi, fv]):
            wf_type, wf_comment = classify_waveform(psv, edv, pi, fv)
            st.markdown(f"**波形分類:** {wf_type}")
            st.caption(f"説明: {wf_comment}")
        else:
            st.markdown("**波形分類:** 判定不能")
            st.caption("説明: パラメータが不足しています")

    # ===== AI診断コメント =====
    with st.container(border=True):
        with st.expander("🤖 AIによる診断コメントを表示 / 非表示"):
            if st.button("AI診断を実行"):
                tav = form.get("tav", 999)
                edv = form.get("edv", 999)
                ri = form.get("ri", 0)
                pi = form.get("pi", 0)
                fv = form.get("fv", 9999)
                tamv = form.get("tamv", 1)

                ai_main_comment = ""
                ai_supplement = []

                if tav < 34.5 and edv < 40.4 and ri >= 0.68 and pi >= 1.3:
                    ai_main_comment = "TAVとEDVの低下。RIとPIの上昇。早急なVAIVT提案が必要です。急な閉塞の危険性があります。"
                elif tav < 34.5 and pi >= 1.3 and edv < 40.4:
                    ai_main_comment = "TAVおよびEDVの低下に加え、PIが上昇。吻合部近傍の高度狭窄が強く疑われます。VAIVT提案を検討してください"
                elif tav < 34.5 and pi >= 1.3:
                    ai_main_comment = "TAVの低下に加え、PIが上昇。吻合部近傍の高度狭窄が疑われます"
                elif tav < 34.5 and edv < 40.4 and pi < 1.3:
                    ai_main_comment = "TAVとEDVが低下しており、中等度の吻合部狭窄が疑われます"
                elif tav < 34.5 and edv >= 40.4:
                    ai_main_comment = "TAVが低下しており、軽度の吻合部狭窄の可能性があります"
                elif ri >= 0.68 and edv < 40.4:
                    ai_main_comment = "RIが高く、EDVが低下。末梢側の狭窄が疑われます"
                elif ri >= 0.68:
                    ai_main_comment = "RIが上昇しています。末梢抵抗の増加が示唆されますが、他のパラメータ異常がないため再検が必要です"
                elif fv < 500:
                    ai_main_comment = "血流量がやや低下しています。経過観察が望まれますが、他のパラメータ異常がないため再検が必要です"
                elif score == 0:
                    ai_main_comment = "正常だと思います。経過観察お願いします"
                else:
                    ai_main_comment = "特記すべき高度な異常所見は検出されませんでしたが、一部パラメータに変化が見られます"

                if tav < 25 and 500 <= fv <= 1000:
                    ai_supplement.append("TAVが非常に低く、FVは正常範囲 → 上腕動脈径が大きいため、過大評価の可能性があります")
                if fv > 1500:
                    ai_supplement.append("FVが高値です。large shuntの可能性があります。身体症状の確認が必要です。")
                if ri >= 0.68 and pi >= 1.3 and fv >= 400 and tav >= 50:
                    ai_supplement.append("RI・PIが上昇していますが、FV・TAVは正常値です。吻合部近傍の分岐血管が影響している可能性があります。遮断試験を実施してください。")

                st.subheader("🧠 AI診断コメント")
                st.info(ai_main_comment)
                for sup in ai_supplement:
                    st.info(sup)

