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
        # 複数候補があれば最大のものを返す
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
    # ROI を左右と縦方向に拡張して数値が映る範囲を広げる
    roi = img_cv[int(h * 0.05):int(h * 0.6), int(w * 0.01):int(w * 0.5)]
    results = reader.readtext(roi)

    lines = [(text.strip(), conf, bbox) for bbox, text, conf in results if conf > 0.4]
    keywords = KEYWORDS_BY_MANUFACTURER[manufacturer]
    extracted = {}
    used_indices = set()

    # --- 横方向パターン（同行にラベルと数値） ---
    for idx, (text, conf, bbox) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in text.lower() for kw in variations):
                value = extract_number(text)
                if value is not None:
                    extracted[key] = value
                    used_indices.add(idx)

    # --- 縦方向補助パターン（ラベルのすぐ下の行に数値があることを仮定） ---
    for idx, (text, conf, bbox) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in text.lower() for kw in variations):
                # 下の行を探す
                j = idx + 1
                if j < len(lines) and j not in used_indices:
                    _, val_text, _ = lines[j]
                    value = extract_number(val_text)
                    if value is not None:
                        extracted[key] = value
                        used_indices.add(j)

    return extracted, results

def classify_waveform(psv, edv, pi, fv):
    # 分類ルール（強化版）
    if edv < 5 and fv < 100:
        return "Type V", "閉塞型（Ⅴ型）：EDVほぼゼロ・流量非常に低い"
    elif fv > 1500:
        return "Type I", "過大血流型（Ⅰ型）：FVが1500以上"
    elif pi >= 1.3 and edv < 40.4:
        # PI高 + EDV低：末梢抵抗 or 狭窄傾向強め
        return "Type IV", "末梢狭窄型（Ⅳ型）：PI高値、EDV低下傾向"
    elif pi >= 1.3:
        # PI高でも EDVが保たれていたらⅢに近づける
        return "Type III", "中等度狭窄型（Ⅲ型）：PI高値＋切痕傾向"
    elif fv < 500 and edv < 40.4:
        # 血流量低下 + 中等度低 EDV → 狭窄傾向（Type IV寄り）
        return "Type IV", "末梢狭窄型（Ⅳ型）：FV低下 × EDV低下"
    else:
        # それ以外は II 型扱い
        return "Type II", "良好波形型（Ⅱ型）：EDV保たれ、PI正常域"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="シャントOCR", layout="centered")
st.title("🩺 シャント画像の数値自動抽出＆診断")

st.sidebar.title("⚙️ メーカー設定")
manufacturer = st.sidebar.selectbox(
    "画像のメーカーを選択してください",
    ["GEヘルスケア", "FUJIFILM", "コミカミノルタ"]
)

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

    # ----- 自動評価セクション -----
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
            if level == "warning":
                st.warning(f"- {comment}")
            else:
                st.write(f"- {comment}")

    # ----- AI診断コメント -----
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
                    ai_main_comment = "TAVとEDVの低下。RIとPIの上昇。早急なVAIVT提案が必要です。閉塞リスク高。"
                elif tav < 34.5 and pi >= 1.3 and edv < 40.4:
                    ai_main_comment = "TAV低下＋EDV低下＋PI上昇。吻合部近傍の高度狭窄が疑われます。VAIVT検討を。"
                elif tav < 34.5 and pi >= 1.3:
                    ai_main_comment = "TAV低下＋PI上昇。高度狭窄疑い。"
                elif tav < 34.5 and edv >= 40.4:
                    ai_main_comment = "TAV低下あり。軽度狭窄可能性。"
                elif ri >= 0.68 and edv < 40.4:
                    ai_main_comment = "RI高値＋EDV低下。末梢側狭窄疑い。"
                elif ri >= 0.68:
                    ai_main_comment = "RI上昇。末梢抵抗増加の可能性。"
                elif fv < 500:
                    ai_main_comment = "血流量やや低下。追加評価検討。"
                elif score == 0:
                    ai_main_comment = "正常値域。経過観察推奨。"
                else:
                    ai_main_comment = "明確な異常所見なし。ただし一部値に変化。"

                if tav < 25 and 500 <= fv <= 1000:
                    ai_supplement.append("TAV非常に低値、FV正常圏内 → 血管径の影響注意")
                if fv > 1500:
                    ai_supplement.append("FV高値：large shunt の可能性")
                if ri >= 0.68 and pi >= 1.3 and fv >= 400 and tav >= 50:
                    ai_supplement.append("RI・PI高値だが FV/TAV 正常値 → 分岐血管影響可能")

                st.info(f"🧠 主コメント: {ai_main_comment}")
                if ai_supplement:
                    st.write("#### 💬 補足コメント")
                    for sup in ai_supplement:
                        st.write(f"- {sup}")

    # ----- 波形分類表示 -----
    st.subheader("📈 波形分類結果")
    with st.expander("📊 波形分類と説明（クリックで展開）"):
        wf_type, wf_comment = classify_waveform(psv, edv, pi, fv)
        st.markdown(f"**波形分類:** {wf_type}")
        st.caption(f"説明: {wf_comment}")
