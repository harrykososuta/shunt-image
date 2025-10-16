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
        "TAMV": ["TAMAX", "TA MAX"],
        "TAV": ["TAMEAN", "TA MEAN"],
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
    full_text = " ".join([t for t, _ in lines])

    for key, variations in keywords.items():
        for label, conf in lines:
            if any(kw.lower() in label.lower() for kw in variations):
                value = extract_number(label)
                if value is not None:
                    extracted[key] = value
                    break

    for i in range(len(lines) - 1):
        label, _ = lines[i]
        value_line, _ = lines[i + 1]
        for key, variations in keywords.items():
            if any(kw.lower() in label.lower() for kw in variations):
                value = extract_number(value_line)
                if value is not None:
                    extracted[key] = value

    if "PI" not in extracted:
        m = re.search(r"PI\s*[:=]?\s*(\d+\.\d+)", full_text)
        if m:
            extracted["PI"] = float(m.group(1))

    if "TAMV" in extracted and "TAV" in extracted:
        if extracted["TAMV"] == extracted["TAV"]:
            extracted["TAV"] = extracted["TAMV"] * 0.7

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

    # 自動評価スコア
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

    # 波形分類
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

    # ===== AI診断コメントセクション =====
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

        note = st.text_area("備考（自由記述）", placeholder="観察メモや特記事項などがあれば記入")

        with st.expander("📌 追加情報を表示"):
            TAVR = tav / tamv if tamv != 0 else 0
            RI_PI = ri / pi if pi != 0 else 0

            st.write("### TAVRの算出")
            st.write(f"TAVR: {TAVR:.2f}")
            st.write("### RI/PI の算出")
            st.write(f"RI/PI: {RI_PI:.2f}")

            st.write("### 波形分類")
            st.markdown("""
            - Ⅰ・Ⅱ型：シャント機能は問題なし  
            - Ⅲ型：50％程度の狭窄があるため精査  
            - Ⅳ型：VAIVT提案念頭に精査  
            - Ⅴ型：シャント閉塞の可能性大
            """)

            st.write("### 追加コメント")
            st.markdown("吻合部付近に2.0mmを超える分岐血管がある場合は遮断試験を行ってください")
            st.write("### 補足コメント")
            st.markdown("この補足は評価に必要な周辺知識を補完するものです。※検査時の注意点などをここにまとめられます")
