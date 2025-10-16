import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
import re
from collections import OrderedDict

reader = easyocr.Reader(['en'])

# =============================
# OCR処理関連
# =============================

def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def extract_number(text):
    """数値抽出の安定化"""
    if not isinstance(text, str):
        text = str(text)
    matches = re.findall(r"\d+\.\d+", text.replace(",", ""))
    if matches:
        return float(matches[0])
    return None

KEYWORDS_BY_MANUFACTURER = {
    "GEヘルスケア": {
        "PSV": ["PS", "P5", "PSV"],
        "EDV": ["ED", "EDV"],
        "TAMV": ["TAMAX", "TA MAX"],  # 最大速度
        "TAV": ["TAMEAN", "TA MEAN"], # 平均速度
        "PI": ["PI"],
        "RI": ["RI"],
        "FV": ["FV"],
        "VF_Diam": ["VF Diam", "VF", "VFD"]
    },
    "FUJIFILM": {
        "PSV": ["PSV"],
        "EDV": ["Ved"],
        "TAMV": ["TAP"],
        "TAV": ["TAM"],
        "PI": ["PI"],
        "RI": ["RI"],
        "FV": ["VF"],
        "VF_Diam": ["VF Diam", "VF", "VFD"]
    },
    "コミカミノルタ": {
        "PSV": ["PSV"],
        "EDV": ["Ved"],
        "TAMV": ["Vm-peak"],
        "TAV": ["Vm-mean"],
        "PI": ["PI"],
        "RI": ["RI"],
        "FV": ["FVol"],
        "VF_Diam": ["VF Diam", "VF", "VFD"]
    }
}


def extract_parameters(img_pil, manufacturer):
    img_cv = pil_to_cv(img_pil)
    h, w = img_cv.shape[:2]
    roi = img_cv[int(h * 0.05):int(h * 0.55), int(w * 0.02):int(w * 0.45)]

    results = reader.readtext(roi)
    lines = [(text.strip(), conf) for _, text, conf in results if conf > 0.3]

    keywords = KEYWORDS_BY_MANUFACTURER[manufacturer]
    extracted = {}

    # ---- ラベル単体検出の補助用全文 ----
    full_text = " ".join([t for t, _ in lines])

    for key, variations in keywords.items():
        for label, conf in lines:
            if any(kw.lower() in label.lower() for kw in variations):
                value = extract_number(label)
                if value is not None:
                    extracted[key] = value
                    break

    # ---- ラベル＋値ペアを補完 ----
    for i in range(len(lines) - 1):
        label, _ = lines[i]
        value_line, _ = lines[i + 1]
        for key, variations in keywords.items():
            if any(kw.lower() in label.lower() for kw in variations):
                value = extract_number(value_line)
                if value is not None:
                    extracted[key] = value

    # ---- PI 補完 (全体文字列から直接拾う) ----
    if "PI" not in extracted:
        m = re.search(r"PI\s*[:=]?\s*(\d+\.\d+)", full_text)
        if m:
            extracted["PI"] = float(m.group(1))

    # ---- TAV と TAMV の混同を避ける補正 ----
    if "TAMV" in extracted and "TAV" in extracted:
        if extracted["TAMV"] == extracted["TAV"]:
            extracted["TAV"] = extracted["TAMV"] * 0.7  # 平均速度補正（経験的）

    # ---- 表示順を整える ----
    ordered = OrderedDict()
    for key in ["PSV", "EDV", "TAMV", "TAV", "RI", "PI", "FV", "VF_Diam"]:
        if key in extracted:
            ordered[key] = extracted[key]

    return ordered, results


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="シャントOCR", layout="centered")
st.title("🩺 シャント画像の数値自動抽出＆診断")

st.sidebar.title("⚙️ メーカー設定")
manufacturer = st.sidebar.selectbox("画像のメーカーを選択してください", 
                                    ["GEヘルスケア", "FUJIFILM", "コミカミノルタ"])

uploaded = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="入力画像", use_container_width=True)

    with st.spinner("OCR解析中..."):
        params, raw = extract_parameters(img, manufacturer)

    st.subheader("📊 抽出されたパラメータ")
    if params:
        st.json(params)
    else:
        st.warning("パラメータが見つかりませんでした")

    # ===== 自動評価セクション =====
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
