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
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
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
    roi = img_cv[int(h * 0.05):int(h * 0.6), int(w * 0.01):int(w * 0.5)]
    results = reader.readtext(roi)

    lines = [(bbox, text.strip(), conf) for bbox, text, conf in results if conf > 0.4]
    keywords = KEYWORDS_BY_MANUFACTURER[manufacturer]
    extracted = {}
    used_labels = set()

    for i, (bbox_i, text_i, conf_i) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in text_i.lower() for kw in variations):
                candidate = None
                best_dist = None
                for j, (bbox_j, text_j, conf_j) in enumerate(lines):
                    if j == i:
                        continue
                    val = extract_number(text_j)
                    if val is None:
                        continue
                    cx_i = np.mean([pt[0] for pt in bbox_i])
                    cy_i = np.mean([pt[1] for pt in bbox_i])
                    cx_j = np.mean([pt[0] for pt in bbox_j])
                    cy_j = np.mean([pt[1] for pt in bbox_j])
                    dist = abs(cy_j - cy_i) + abs(cx_j - cx_i) * 0.5
                    if best_dist is None or dist < best_dist:
                        candidate = val
                        best_dist = dist
                if candidate is not None:
                    extracted[key] = candidate
                    used_labels.add(i)

    # 横並びと縦並び補助
    used_indices = set()
    for idx, (bbox, text, conf) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in text.lower() for kw in variations):
                value = extract_number(text)
                if value is not None and key not in extracted:
                    extracted[key] = value
                    used_indices.add(idx)

    for idx, (bbox, label_text, conf) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in label_text.lower() for kw in variations):
                j = idx + 1
                if j < len(lines) and j not in used_indices:
                    _, val_text, _ = lines[j]
                    value = extract_number(val_text)
                    if value is not None and key not in extracted:
                        extracted[key] = value
                        used_indices.add(j)

    return extracted, results

def classify_waveform(psv, edv, pi, fv):
    if edv < 5 and fv < 100:
        return "Type V", "閉塞型（Ⅴ型）：EDVほぼゼロ・流量非常に低い"
    elif fv > 1500:
        return "Type I", "過大血流型（Ⅰ型）：FVが1500以上"
    elif pi >= 1.3 and edv < 40.4:
        return "Type IV", "末梢狭窄型（Ⅳ型）：PI高値、EDV低下傾向"
    elif pi >= 1.3:
        return "Type III", "中等度狭窄型（Ⅲ型）：PI高値＋切痕傾向"
    elif fv < 500 and edv < 40.4:
        return "Type IV", "末梢狭窄型（Ⅳ型）：FV低下 × EDV低下"
    else:
        return "Type II", "良好波形型（Ⅱ型）：EDV保たれ、PI正常域"

# ========== Streamlit UI ==========
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

    # -------- 自動評価コード --------
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

    # -------- AI診断コメント --------
    tav = form.get("tav", 0)
    tamv = form.get("tamv", 1)
    ri = form.get("ri", 0)
    pi = form.get("pi", 0.1)
    fv = form.get("fv", 0)
    edv = form.get("edv", 0)

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
                    ai_main_comment = "TAVとEDVの低下 + PI上昇 → 高度吻合部狭窄が疑われます。"
                elif tav < 34.5 and pi >= 1.3:
                    ai_main_comment = "TAVの低下 + PI上昇 → 吻合部近傍の高度狭窄が疑われます"
                elif tav < 34.5 and edv < 40.4:
                    ai_main_comment = "TAVとEDVが低下しており、中等度の吻合部狭窄が疑われます"
                elif ri >= 0.68 and edv < 40.4:
                    ai_main_comment = "RI高値 × EDV低下 → 末梢側狭窄が疑われます"
                elif score == 0:
                    ai_main_comment = "正常です。経過観察を推奨します。"
                else:
                    ai_main_comment = "一部パラメータに変化あり。再評価が望まれます。"

                st.info(f"🧠 主コメント: {ai_main_comment}")

                if tav < 25 and 500 <= fv <= 1000:
                    ai_supplement.append("TAVが非常に低いがFV正常 → 過大評価の可能性あり")
                if fv > 1500:
                    ai_supplement.append("FV高値 → large shunt の可能性あり")
                if ri >= 0.68 and pi >= 1.3 and fv >= 400 and tav >= 50:
                    ai_supplement.append("RI・PI高いがFV・TAV正常 → 分岐血管かも。遮断試験検討")

                if ai_supplement:
                    st.write("#### 💬 補足コメント")
                    for sup in ai_supplement:
                        st.write(f"- {sup}")

    # -------- 波形分類セクション --------
    st.subheader("📈 波形分類結果")
    with st.expander("📊 波形分類と説明（クリックで展開）"):
        psv = params.get("PSV") or params.get("psv") or 0
        edv = params.get("EDV") or params.get("edv") or 0
        pi = params.get("PI") or params.get("pi") or 0
        fv = params.get("FV") or params.get("fv") or 0

        if all([psv, edv, pi, fv]):
            wf_type, wf_comment = classify_waveform(psv, edv, pi, fv)
            st.markdown(f"**波形分類:** {wf_type}")
            st.caption(f"説明: {wf_comment}")
        else:
            st.markdown("**波形分類:** 判定不能")
            st.caption("説明: 必要なパラメータが欠損しているため分類できません")
