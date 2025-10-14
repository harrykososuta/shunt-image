import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
import re

reader = easyocr.Reader(['en'])

def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def extract_number(text):
    matches = re.findall(r"\d+\.\d+", text.replace(",", ""))
    if matches:
        return float(max(matches, key=lambda x: float(x)))
    return None

KEYWORDS_BY_MANUFACTURER = {
    "GEヘルスケア": {
        "PSV": ["PS", "P5", "PSV"],
        "EDV": ["ED", "EDV"],
        "TAMV": ["TAMAX", "TA MAX"],
        "TAV": ["TAMEAN", "TAV"],
        "PI": ["PI"],
        "RI": ["RI"],
        "FV": ["FV"],
        "VF_Diam": ["VF Diam", "VF", "VFD"]
    },
    "FUJIFILM": {
        "PSV": ["PS", "P5", "PSV"],
        "EDV": ["Ved"],
        "TAMV": ["TAP"],
        "TAV": ["TAM"],
        "PI": ["PI"],
        "RI": ["RI"],
        "FV": ["VF"],
        "VF_Diam": ["VF Diam", "VF", "VFD"]
    },
    "コミカミノルタ": {
        "PSV": ["PS", "P5", "PSV"],
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
    roi = img_cv[int(h * 0.05):int(h * 0.6), int(w * 0.0):int(w * 0.35)]
    results = reader.readtext(roi)

    lines = [(text.strip(), conf) for _, text, conf in results if conf > 0.4]
    keywords = KEYWORDS_BY_MANUFACTURER[manufacturer]
    extracted = {}
    used_indices = set()

    for idx, (text, conf) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in text.lower() for kw in variations):
                value = extract_number(text)
                if value is not None:
                    extracted[key] = value
                    used_indices.add(idx)

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

def classify_waveform(psv, edv, pi, fv):
    if edv < 5 and fv < 500:
        return "Type V", "閉塞型（V型）：EDVほぼゼロ・流量非常に低"
    elif fv > 1500:
        return "Type I", "過大血流型（I型）：FVが1500ml/min以上"
    elif pi > 1.3 and edv < 40:
        return "Type III", "狭窄型（III型）：PI高値かつEDV低下"
    elif pi < 1.3 and edv < 40:
        return "Type II", "中等度狭窄型（II型）：EDV低下"
    elif pi > 1.3 and edv >= 40:
        return "Type IV", "高抵抗型（IV型）：PI高値だがEDVは保たれる"
    else:
        return "判定不能", "波形分類の基準を満たしません。再評価してください"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="シャントOCR", layout="centered")
st.title("🩺 シャント画像の数値自動抽出＆診断")

st.sidebar.title("⚙️ メーカー設定")
manufacturer = st.sidebar.selectbox("画像のメーカーを選択してください", ["GEヘルスケア", "FUJIFILM", "コミカミノルタ"])

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
        st.error("🔴 高リスク：専門的な評価が必要です")

    if comments:
        st.write("### 評価コメント")
        for level, comment in comments:
            st.warning(f"- {comment}")

    # --- AI診断コメント ---
    tav = form.get("tav", 0)
    tamv = form.get("tamv", 1)
    ri = form.get("ri", 0)
    pi = form.get("pi", 0.1)
    fv = form.get("fv", 0)
    edv = form.get("edv", 0)
    psv = form.get("psv", 0)

    TAVR = tav / tamv if tamv else 0
    RI_PI = ri / pi if pi else 0

    with st.container(border=True):
        with st.expander("🤖 AIによる診断コメントを表示 / 非表示"):
            if st.button("AI診断を実行"):
                ai_main_comment = ""
                ai_supplement = []

                if tav < 34.5 and edv < 40.4 and ri >= 0.68 and pi >= 1.3:
                    ai_main_comment = "TAVとEDVの低下。RIとPIの上昇。早急なVAIVT提案が必要です。"
                elif tav < 34.5 and pi >= 1.3 and edv < 40.4:
                    ai_main_comment = "TAVとEDV低下＋PI高値 → 吻合部近傍の高度狭窄が疑われます。"
                elif tav < 34.5 and pi >= 1.3:
                    ai_main_comment = "TAV低下＋PI高値 → 高度狭窄の疑い"
                elif tav < 34.5 and edv >= 40.4:
                    ai_main_comment = "TAVが低下 → 軽度狭窄の可能性"
                elif ri >= 0.68 and edv < 40.4:
                    ai_main_comment = "RI高値＋EDV低下 → 末梢側狭窄が疑われます"
                elif score == 0:
                    ai_main_comment = "正常と考えられます。経過観察を推奨します。"
                else:
                    ai_main_comment = "一部異常所見あり。追加検査をご検討ください。"

                if fv > 1500:
                    ai_supplement.append("FVが高値 → large shuntの可能性あり")

                st.info(f"🧠 主コメント: {ai_main_comment}")
                if ai_supplement:
                    st.write("#### 💬 補足コメント")
                    for sup in ai_supplement:
                        st.write(f"- {sup}")

    # --- 波形分類表示 ---
    st.subheader("📈 波形分類結果")
    with st.expander("🧬 波形分類と説明（クリックで展開）"):
        waveform_type, waveform_comment = classify_waveform(psv, edv, pi, fv)
        st.write(f"**波形分類**: {waveform_type}")
        st.caption(f"説明: {waveform_comment}")
