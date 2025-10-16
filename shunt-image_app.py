import streamlit as st
import easyocr
import numpy as np
import re
from PIL import Image
import cv2

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

    # Debug 全体出力
    st.write("OCR raw:", results)

    lines = [(bbox, text.strip(), conf) for bbox, text, conf in results if conf > 0.2]
    keywords = KEYWORDS_BY_MANUFACTURER[manufacturer]
    extracted = {}
    used_labels = set()

    # ラベル → 値マッチング
    for i, (bbox_i, text_i, conf_i) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in text_i.lower() for kw in variations):
                best_val = None
                best_score = None
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
                    # 距離スコア（縦優先重み強め）
                    score = abs(cy_j - cy_i) * 2 + abs(cx_j - cx_i)
                    # 単位補正：text_j に “ml/min” や “cm/s” があればスコア優遇
                    if "ml/min" in text_j:
                        if key == "FV":
                            score *= 0.5
                    if "cm/s" in text_j:
                        if key in ("PSV", "EDV", "TAV", "TAMV"):
                            score *= 0.5
                    # 範囲チェック
                    if key == "RI" and not (0 <= val <= 5):
                        continue
                    if best_score is None or score < best_score:
                        best_score = score
                        best_val = val
                if best_val is not None:
                    extracted[key] = best_val
                    used_labels.add(i)

    # フォールバック横／縦補助
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

    # バックアップ方式：raw テキスト結合で “PS 93.48” 等を探す
    full_text = " ".join([text for (_, text, _) in lines])
    # PSV パターン
    m = re.search(r"PS\s*(\d+\.\d+)", full_text)
    if m and "PSV" not in extracted:
        extracted["PSV"] = float(m.group(1))
    m2 = re.search(r"ED\s*(\d+\.\d+)", full_text)
    if m2 and "EDV" not in extracted:
        extracted["EDV"] = float(m2.group(1))
    m3 = re.search(r"FV\s*(\d+\.\d+)", full_text)
    if m3 and "FV" not in extracted:
        extracted["FV"] = float(m3.group(1))

    return extracted, results

def classify_waveform(psv, edv, pi, fv):
    if edv < 5 and fv < 100:
        return "Type V", "閉塞型（Ⅴ型）：EDV ≒ 0, 流量非常に低"
    elif fv > 1500:
        return "Type I", "過大血流型（Ⅰ型）：FV が大きい"
    elif pi >= 1.3 and edv < 40.4:
        return "Type IV", "末梢狭窄型（Ⅳ型）：PI 高め、EDV 低め"
    elif pi >= 1.3:
        return "Type III", "狭窄型（Ⅲ型）：PI 高め"
    elif fv < 500 and edv < 40.4:
        return "Type IV", "末梢狭窄型（Ⅳ型）：FV 低め & EDV 低め"
    else:
        return "Type II", "中等型（Ⅱ型）：EDV 保たれ、PI 普通"

# UI 部分
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

    # 自動評価
    st.subheader("🔍 自動評価スコア")
    form = {k.lower(): v for k, v in params.items()}
    score = 0
    comments = []
    if form.get("tav", 999) <= 34.5:
        score += 1
        comments.append(("warning", "TAV が 34.5 cm/s 以下 → 低血流疑い"))
    if form.get("ri", 0) >= 0.68:
        score += 1
        comments.append(("warning", "RI が 0.68 以上 → 高抵抗疑い"))
    if form.get("pi", 0) >= 1.3:
        score += 1
        comments.append(("warning", "PI が 1.3 以上 → 波形異常"))
    if form.get("edv", 999) <= 40.4:
        score += 1
        comments.append(("warning", "EDV が 40.4 cm/s 以下 → 拡張期血流低下"))

    st.write(f"評価スコア: {score} / 4")
    if score == 0:
        st.success("🟢 正常：経過観察推奨")
    elif score in [1,2]:
        st.warning("🟡 要注意：追加評価必要")
    else:
        st.error("🔴 高リスク：専門評価必要")

    if comments:
        st.write("### 評価コメント")
        for level, comment in comments:
            if level == "warning":
                st.warning(f"- {comment}")

    # AI診断
    tav = form.get("tav", 0)
    tamv = form.get("tamv", 1)
    ri = form.get("ri", 0)
    pi = form.get("pi", 0.1)
    fv = form.get("fv", 0)
    edv = form.get("edv", 0)

    with st.container(border=True):
        with st.expander("🤖 AI診断コメント"):
            if st.button("AI診断を実行"):
                ai_main = ""
                supplement = []
                if tav < 34.5 and edv < 40.4 and ri >= 0.68 and pi >= 1.3:
                    ai_main = "TAV, EDV 低下、RI, PI 上昇あり。VAIVT を強く検討。"
                elif tav < 34.5 and pi >= 1.3 and edv < 40.4:
                    ai_main = "TAV, EDV 低下 + PI 上昇 → 高度狭窄疑い"
                elif tav < 34.5 and pi >= 1.3:
                    ai_main = "TAV 低下 + PI 上昇 → 狭窄疑い"
                elif tav < 34.5 and edv < 40.4:
                    ai_main = "TAV・EDV 低下 → 中等度狭窄疑い"
                elif ri >= 0.68 and edv < 40.4:
                    ai_main = "RI 高値 + EDV 低下 → 末梢狭窄疑い"
                elif score == 0:
                    ai_main = "正常範囲と判断"
                else:
                    ai_main = "明確な高度異常なし。ただし変動あり"

                st.info(f"🧠 主コメント: {ai_main}")

                if tav < 25 and 500 <= fv <= 1000:
                    supplement.append("TAV 極めて低値、FV 正常 → 血管径の影響かも")
                if fv > 1500:
                    supplement.append("FV 高値 → large shunt 可能性")
                if ri >= 0.68 and pi >= 1.3 and fv >= 400 and tav >= 50:
                    supplement.append("RI・PI 高値だが TAV/FV 正常 → 分岐血管影響可能")

                if supplement:
                    st.write("#### 補足コメント")
                    for s in supplement:
                        st.write(f"- {s}")

    # 波形分類
    st.subheader("📈 波形分類結果")
    with st.expander("📊 波形分類と説明"):
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
            st.caption("説明: 必要なパラメータが欠けている可能性があります")
