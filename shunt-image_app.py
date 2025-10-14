import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
import re

# OCRモデルの初期化（英語のみ）
reader = easyocr.Reader(['en'])

def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def extract_number(text):
    match = re.search(r"(\d+\.\d+)", text.replace(",", ""))
    return float(match.group(1)) if match else None

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
    roi = img_cv[int(h * 0.1):int(h * 0.5), int(w * 0.02):int(w * 0.3)]
    results = reader.readtext(roi)

    lines = [(text.strip(), conf) for _, text, conf in results if conf > 0.4]
    keywords = KEYWORDS_BY_MANUFACTURER[manufacturer]
    extracted = {}
    used_indices = set()

    # 1行ラベル＋数値パターン
    for idx, (text, conf) in enumerate(lines):
        for key, variations in keywords.items():
            if any(kw.lower() in text.lower() for kw in variations):
                value = extract_number(text)
                if value is not None:
                    extracted[key] = value
                    used_indices.add(idx)

    # 2行分割パターン
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

def waveform_classification(params, evaluation_result):
    """
    params: dict with keys "psv","edv","pi","ri","fv" (lowercase)
    evaluation_result: dict or tuple indicating whether VAIVT is recommended etc.
    Returns: (wave_type_str, explanation_str)
    """
    psv = params.get("psv", 0)
    edv = params.get("edv", 0)
    pi = params.get("pi", 0)
    ri = params.get("ri", 0)
    fv = params.get("fv", 0)
    # 比率安全ガード
    ratio = edv / psv if psv else 0

    # 評価結果との矛盾チェック用フラグ
    # 例：evaluation_result["vaivt_needed"] が True の場合、Ⅱ型はおかしいなど
    vaivt_needed = evaluation_result.get("vaivt_needed", False)

    # 基本分類ロジック
    # Ⅰ → 過大血流
    if fv >= 1500:
        wave = "Type I"
        expl = "過大血流型：FVが1500以上"
    else:
        # Ⅱ型の候補条件
        cond_ii = (ratio > 0.4 and pi < 1.3 and not vaivt_needed)
        # Ⅲ型条件：切痕傾向 + PI 高値 + EDV比低
        cond_iii = (ratio < 0.4 and pi >= 1.3)
        # Ⅳ型条件：EDV より著しく低下 + PI 高 + 切痕明瞭
        cond_iv = (edv < 30 and pi >= 1.3)
        # Ⅴ型条件：EDV ≈ 0 に近く、流量非常に低い
        cond_v = (edv < 5 or fv < 50)

        if cond_ii:
            wave = "Type II"
            expl = "良好波形型（Ⅱ型）：EDV比高く、PIも正常域"
        elif cond_iii:
            wave = "Type III"
            expl = "中等度狭窄型（Ⅲ型）：PIがやや上昇、EDV低め"
        elif cond_iv:
            wave = "Type IV"
            expl = "高度狭窄型（Ⅳ型）：EDVが著しく低く、PI高"
        elif cond_v:
            wave = "Type V"
            expl = "閉塞型（Ⅴ型）：EDVほぼゼロ・流量非常に低"
        else:
            wave = "Uncertain / 混合型"
            expl = "分類しきれないグレーゾーン"

    # もし VAIVT 要の評価と分類が矛盾するなら警告
    if vaivt_needed and wave in ("Type I", "Type II"):
        expl += "（⚠ VAIVT 提案と矛盾する分類）"

    return wave, expl

def evaluate_params(params):
    """
    あなたの既存の評価コードをここに統合して
    vaivt_needed: True/False などを出すようにする
    """
    # 小文字キーに変換
    f = {k.lower(): v for k, v in params.items()}
    score = 0
    comments = []
    vaivt_needed = False

    if f.get("tav", 0) <= 34.5:
        score += 1
        comments.append(("warning", "TAV が 34.5 cm/s 以下 → 低血流が疑われる"))
    if f.get("ri", 0) >= 0.68:
        score += 1
        comments.append(("warning", "RI が 0.68 以上 → 高抵抗が疑われる"))
    if f.get("pi", 0) >= 1.3:
        score += 1
        comments.append(("warning", "PI が 1.3 以上 → 脈波指数が高い"))
    if f.get("edv", 999) <= 40.4:
        score += 1
        comments.append(("warning", "EDV が 40.4 cm/s 以下 → 拡張期血流速度が低い"))

    if score >= 3:
        vaivt_needed = True

    # AI 診断コメントも…（省略。前述のものをそのまま使えば良い）
    ai_main = ""
    # …（AIロジックをここに入れる）…

    return {
        "score": score,
        "comments": comments,
        "vaivt_needed": vaivt_needed,
        "ai_main": ai_main
    }

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="シャント OCR + 波形分類", layout="centered")
st.title("🩺 シャント波形分類付き数値抽出アプリ")

st.sidebar.title("⚙️ メーカー設定")
manufacturer = st.sidebar.selectbox(
    "画像のメーカーを選択",
    ["GEヘルスケア", "FUJIFILM", "コミカミノルタ"]
)

uploaded = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="入力画像", use_container_width=True)

    with st.spinner("OCR + 波形分類中..."):
        params, raw = extract_parameters(img, manufacturer)
        evaluation = evaluate_params(params)
        wave_type, wave_expl = waveform_classification(params, evaluation)

    st.subheader("📊 抽出されたパラメータ")
    if params:
        st.json(params)
    else:
        st.warning("パラメータが見つかりませんでした")

    st.subheader("🔍 自動評価スコア")
    st.write(f"評価スコア: {evaluation['score']} / 4")
    if evaluation["score"] == 0:
        st.success("🟢 正常：経過観察が推奨されます")
    elif evaluation["score"] in [1, 2]:
        st.warning("🟡 要注意：追加評価が必要です")
    else:
        st.error("🔴 高リスク：専門的な評価が必要です")

    if evaluation["comments"]:
        st.write("### 評価コメント")
        for level, comment in evaluation["comments"]:
            if level == "warning":
                st.warning(f"- {comment}")
            else:
                st.write(f"- {comment}")

    with st.expander("📈 波形分類結果と解説"):
        st.write(f"**波形分類:** {wave_type}")
        st.write(f"**解説:** {wave_expl}")

    with st.expander("🔍 OCR の生データ（デバッグ用）"):
        for _, text, conf in raw:
            st.write(f"[{conf:.2f}] {text}")
