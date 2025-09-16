# app.py
# -------------------------------------------------------------
# 透析シャント機能評価「画像読込＋判読サポート」v0.2.1
# 変更点:
#  - Tesseract の実体バイナリを自動検出（Streamlit Cloud / ローカル両対応）
#  - 未インストール時はOCRをスキップし、丁寧にガイダンス表示
#  - それ以外は v0.2 と同じ（配置に依存しない全画面OCR→ラベル正規化）
# 実行: streamlit run app.py
# -------------------------------------------------------------

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import re
import math
import shutil

# --- OCR backend (pytesseract) 準備 ---
OCR_AVAILABLE = True
try:
    import pytesseract
    # Streamlit Cloud 等でバイナリの場所を自動検出
    _tess = shutil.which("tesseract")
    if _tess:
        pytesseract.pytesseract.tesseract_cmd = _tess
    else:
        # バイナリが見つからない場合はOCR使用不可として扱う
        OCR_AVAILABLE = False
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="シャント機能評価 画像判読サポート", layout="wide")

# =============== 定数/初期設定 ===============
MAKERS = [
    "GE", "FUJIFILM", "CANON", "日立アロカ",
    "コニカミノルタ", "PHILIPS", "SAMSUNG", "Siemens",
]

# 表示ゆれ ⇔ 標準ラベル
ALIASES = {
    "default": {
        "PSV": ["PSV", "Vp", "Vmax", "S"],
        "EDV": ["EDV", "Vd", "D"],
        "TAV": ["TAV", "Vm", "Mean", "Vmean", "MEAN"],
        "TAMV": ["TAMV", "MV", "Time-Avg Mean Velocity", "TAMEAN", "TAMAX"],
        "FV": ["Flow", "Q", "V_flow", "VolFlow", "BFV", "FlowVolume", "FV"],
        "RI": ["RI", "ResistiveIndex"],
        "PI": ["PI", "PulsatilityIndex"],
        "VF_Diam": ["VF Diam", "VFDiam", "VF_Diam", "直径", "Diameter"],
    },
    "GE": {
        "PSV": ["PS", "PSV"],
        "EDV": ["ED", "EDV"],
        "TAMV": ["TAMAX", "TAMV"],
        "TAV": ["TAMEAN", "TAV", "MEAN"],
        "FV": ["FV"],
        "RI": ["RI"],
        "PI": ["PI"],
        "VF_Diam": ["VF Diam", "VFDiam"],
    },
}

THRESH = {
    "RI_high": 0.68,
    "PI_high": 1.25,
    "FV_low": 380.0,   # mL/min
    "TAVR_low": 0.55,
    "TAVR_high": 0.60,
    "SDratio_high": 4.0,
    "Re_high": 2300.0,
}

WAVE_DESCRIPTIONS = {
    "I":  "高抵抗・三相性に近い。拡張期の逆流あり/低下。末梢抵抗が高い所見。",
    "II": "二相性〜高抵抗寄り。拡張期流速は低め。",
    "III":"単相化し拡張期も一定の順行成分。VA影響で低抵抗化。",
    "IV": "拡張期流速が持続高値。低抵抗パターンが顕著。",
    "V":  "ほぼ等流・平坦化。心周期依存性が乏しく、常時高流指向。",
}

# =============== OCR（位置依存しない抽出） ===============
def _preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    th = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    return th

def _ocr_lines_anywhere(pil_img: Image.Image):
    th = _preprocess_for_ocr(pil_img)
    data = pytesseract.image_to_data(
        th, lang="eng", output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6"
    )
    words = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y = data["left"][i], data["top"][i]
        words.append((y, x, txt))
    words.sort(key=lambda t: (t[0], t[1]))
    lines, buf, last_y = [], [], None
    for y, x, txt in words:
        if last_y is None or abs(y - last_y) <= 10:
            buf.append(txt)
        else:
            lines.append(" ".join(buf))
            buf = [txt]
        last_y = y
    if buf:
        lines.append(" ".join(buf))
    return lines

def _maker_patterns(maker: str):
    m = maker if maker in ALIASES else "default"
    alias = ALIASES[m]
    pats = {}
    for std_key, variants in alias.items():
        pats[std_key] = [r"\b" + re.escape(v).replace(r"\ ", r"\s*") + r"\b" for v in variants]
    return pats

_VALUE = r"(-?\d+(?:\.\d+)?)"

def extract_params_anywhere(pil_img: Image.Image, maker: str):
    """画像全体から、メーカー表記ゆれを標準キーに正規化して抽出。"""
    if not OCR_AVAILABLE:
        # OCRなし運用（手入力のみ）でもアプリが落ちないようにする
        return {
            "PSV": None, "EDV": None, "TAV": None, "TAMV": None,
            "PI": None, "RI": None, "FV": None, "VF_Diam": None
        }, []
    lines = _ocr_lines_anywhere(pil_img)
    pats = _maker_patterns(maker)
    results = {
        "PSV": None, "EDV": None, "TAV": None, "TAMV": None,
        "PI": None, "RI": None, "FV": None, "VF_Diam": None
    }
    for ln in lines:
        for std_key, alias_list in pats.items():
            for alias in alias_list:
                m = re.search(alias + r"\s*[:=]?\s*" + _VALUE + r"(?:\s*(cm/s|ml/min|cm))?",
                              ln, re.IGNORECASE)
                if m and results.get(std_key) is None:
                    try:
                        results[std_key] = float(m.group(1))
                    except Exception:
                        pass
    return results, lines

# =============== 計算ユーティリティ ===============
def normalize_labels(maker: str, input_dict: dict) -> dict:
    aliases = {k: set(v) for k, v in ALIASES.get(maker, ALIASES["default"]).items()}
    std = {"PSV": None, "EDV": None, "TAV": None, "TAMV": None, "FV": None, "RI": None, "PI": None, "VF_Diam": None}
    for key, val in input_dict.items():
        for stdk, names in aliases.items():
            if key in names or key == stdk:
                std[stdk] = val
    return std

def calc_metrics(std: dict) -> dict:
    PSV = std.get("PSV")
    EDV = std.get("EDV")
    TAV = std.get("TAV")
    TAMV = std.get("TAMV") if std.get("TAMV") is not None else TAV
    FV  = std.get("FV")
    RI  = std.get("RI")
    PI  = std.get("PI")

    TAVR = RIPI = SDratio = None

    if TAV is not None and TAMV not in (None, 0):
        TAVR = TAV / TAMV
    if RI is not None and PI not in (None, 0):
        RIPI = RI / PI
    if PSV not in (None, 0) and EDV is not None:
        SDratio = PSV / max(EDV, 1e-9)

    if RI is None and PSV not in (None, 0):
        RI = (PSV - (EDV if EDV is not None else 0.0)) / PSV
    if PI is None and TAMV not in (None, 0) and PSV is not None and EDV is not None:
        PI = (PSV - EDV) / TAMV

    return {"PSV": PSV, "EDV": EDV, "TAV": TAV, "TAMV": TAMV, "FV": FV, "RI": RI, "PI": PI,
            "TAVR": TAVR, "RIPI": RIPI, "SDratio": SDratio}

def reynolds_number(mean_velocity_m_per_s, vessel_diameter_mm, rho=1060.0, mu=0.0035):
    if mean_velocity_m_per_s is None or vessel_diameter_mm is None:
        return None
    D = vessel_diameter_mm / 1000.0
    return (rho * mean_velocity_m_per_s * D) / mu

def suggest_wave_type(ri, pi):
    if ri is None or pi is None:
        return None
    if ri >= 0.8:      return "I"
    if 0.7 <= ri < 0.8:return "II"
    if 0.6 <= ri < 0.7:return "III"
    if 0.5 <= ri < 0.6:return "IV"
    return "V"

def highlight(val, key):
    if val is None:
        return "—"
    abnormal = False
    if key == "FV" and val is not None:
        abnormal = val < THRESH["FV_low"]
    if key == "RI" and val is not None:
        abnormal = val > THRESH["RI_high"]
    if key == "PI" and val is not None:
        abnormal = val > THRESH["PI_high"]
    if key == "TAVR" and val is not None:
        abnormal = (val < THRESH["TAVR_low"]) or (val > THRESH["TAVR_high"])
    if key == "SDratio" and val is not None:
        abnormal = val > THRESH["SDratio_high"]
    if key == "Re" and val is not None:
        abnormal = val > THRESH["Re_high"]
    style = "color: red; font-weight: 700;" if abnormal else ""
    txt = f"{val:.3f}" if isinstance(val, float) else f"{val}"
    return f"<span style='{style}'>{txt}</span>"

# =============== サイドバー ===============
st.sidebar.header("設定")
maker = st.sidebar.selectbox("メーカーを選択", MAKERS, index=0)
st.sidebar.caption("選択メーカーに応じてOCRラベル正規化を行います。")

st.sidebar.subheader("血液物性 / 血管径")
rho = st.sidebar.number_input("血液密度 ρ [kg/m³]", value=1060.0, step=10.0, min_value=900.0, max_value=1200.0)
mu  = st.sidebar.number_input("粘度 μ [Pa·s]", value=0.0035, step=0.0001, min_value=0.0010, max_value=0.02)
diam_mm = st.sidebar.number_input("上腕動脈 直径 D [mm]", value=4.0, step=0.5, min_value=1.0, max_value=12.0)

st.sidebar.subheader("基準値（赤色強調）")
THRESH["RI_high"]      = st.sidebar.number_input("RI 高値しきい値", value=float(THRESH["RI_high"]), step=0.01)
THRESH["PI_high"]      = st.sidebar.number_input("PI 高値しきい値", value=float(THRESH["PI_high"]), step=0.01)
THRESH["FV_low"]       = st.sidebar.number_input("FV 低値しきい値 [mL/min]", value=float(THRESH["FV_low"]), step=10.0)
THRESH["TAVR_low"]     = st.sidebar.number_input("TAVR 低値", value=float(THRESH["TAVR_low"]), step=0.01)
THRESH["TAVR_high"]    = st.sidebar.number_input("TAVR 高値", value=float(THRESH["TAVR_high"]), step=0.01)
THRESH["SDratio_high"] = st.sidebar.number_input("SDratio 高値", value=float(THRESH["SDratio_high"]), step=0.1)
THRESH["Re_high"]      = st.sidebar.number_input("Reynolds 数 乱流目安", value=float(THRESH["Re_high"]), step=50.0)

# =============== メイン ===============
st.header("シャント機能評価：画像読込 & 自動解析（v0.2.1）")

col1, col2 = st.columns([1.2, 1])
with col1:
    uploaded = st.file_uploader("画像をアップロード（JPEG/PNG）", type=["jpg", "jpeg", "png"])
    if not OCR_AVAILABLE:
        st.warning("OCRモジュール（Tesseract）が未インストールのため、画像からの自動抽出は無効です。"
                   "Streamlit Cloud の場合はリポジトリ直下に packages.txt を追加し、"
                   "`tesseract-ocr` と `tesseract-ocr-eng` を記載してください。")

    pil_img = None
    ocr_std = None
    ocr_lines_dbg = None

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="入力画像プレビュー", use_column_width=True)

        if OCR_AVAILABLE:
            with st.spinner("OCR抽出中..."):
                ocr_std, ocr_lines_dbg = extract_params_anywhere(pil_img, maker)
            if any(v is not None for v in ocr_std.values()):
                st.success("OCR抽出（標準キー）")
                st.json(ocr_std)
            else:
                st.info("OCRで有効なパラメータが抽出できませんでした。手入力欄をご利用ください。")

with col2:
    st.subheader("手入力（または OCR 値の上書き）")
    st.caption("メーカー表記で入力 → 下で標準ラベルに自動変換します。空欄は未使用。")
    maker_alias = ALIASES.get(maker, ALIASES["default"])

    inputs = {}
    for std_key, alist in maker_alias.items():
        show_name = alist[0] if len(alist) else std_key
        inputs[show_name] = None

    if OCR_AVAILABLE and uploaded is not None and ocr_std:
        for std_key, val in ocr_std.items():
            if val is None:
                continue
            if std_key in maker_alias:
                rep = maker_alias[std_key][0] if maker_alias[std_key] else std_key
                inputs[rep] = val

    for show_name, init_val in inputs.items():
        s = "" if init_val is None else str(init_val)
        val_str = st.text_input(f"{show_name}", value=s)
        try:
            inputs[show_name] = float(val_str) if val_str.strip() != "" else None
        except Exception:
            inputs[show_name] = None

# =============== 標準化＆計算 ===============
std      = normalize_labels(maker, inputs)
derived  = calc_metrics(std)
mean_vel = derived.get("TAV") if derived.get("TAV") not in (None, 0) else derived.get("TAMV")
Re       = reynolds_number(mean_vel, vessel_diameter_mm=diam_mm, rho=rho, mu=mu)

# =============== 表示 ===============
st.subheader("標準化パラメータ & 自動算出")

cols = st.columns(4)
for i, k in enumerate(["PSV","EDV","TAV","TAMV"]):
    with cols[i % 4]:
        st.markdown(f"**{k}**: {highlight(derived.get(k), k)}", unsafe_allow_html=True)

cols = st.columns(4)
for i, k in enumerate(["FV","RI","PI","TAVR"]):
    with cols[i % 4]:
        st.markdown(f"**{k}**: {highlight(derived.get(k), k)}", unsafe_allow_html=True)

cols = st.columns(3)
for i, k in enumerate(["RIPI","SDratio"]):
    with cols[i % 3]:
        st.markdown(f"**{k}**: {highlight(derived.get(k), k)}", unsafe_allow_html=True)

st.markdown(f"**Re（Reynolds数）**: {highlight(Re, 'Re')}", unsafe_allow_html=True)

# =============== 波形タイプ（I〜V） ===============
st.subheader("上腕動脈血流波形タイプ（I〜V）")
def suggest_wave_type_local(ri, pi):
    if ri is None or pi is None:
        return None
    if ri >= 0.8:      return "I"
    if 0.7 <= ri < 0.8:return "II"
    if 0.6 <= ri < 0.7:return "III"
    if 0.5 <= ri < 0.6:return "IV"
    return "V"

suggested = suggest_wave_type_local(derived.get("RI"), derived.get("PI"))
colw1, colw2 = st.columns([1,2])
with colw1:
    wave_type = st.selectbox(
        "自動推定（編集可）",
        options=["—","I","II","III","IV","V"],
        index=(["—","I","II","III","IV","V"].index(suggested) if suggested in ["I","II","III","IV","V"] else 0)
    )
with colw2:
    if wave_type in WAVE_DESCRIPTIONS:
        st.info(f"説明: {WAVE_DESCRIPTIONS[wave_type]}")
    else:
        st.info("現状はRI/PIの暫定境界で推定。後で文献ルールに置換します。")

# =============== 最終コメント（雛形） ===============
with st.expander("最終コメント（雛形）"):
    lines = []
    if derived.get("RI") is not None and derived["RI"] > THRESH["RI_high"]:
        lines.append(f"RIは高値（{derived['RI']:.2f}）で、末梢抵抗上昇が示唆されます。")
    if derived.get("PI") is not None and derived["PI"] > THRESH["PI_high"]:
        lines.append(f"PIは高値（{derived['PI']:.2f}）。波形の脈動性増大を示唆します。")
    if derived.get("FV") is not None and derived["FV"] < THRESH["FV_low"]:
        lines.append(f"流量（FV）は低値（{derived['FV']:.0f} mL/min）。吻合部狭小化や静脈側の抵抗増大に留意。")
    if derived.get("TAVR") is not None and (derived["TAVR"] < THRESH["TAVR_low"] or derived["TAVR"] > THRESH["TAVR_high"]):
        lines.append(f"TAVRは{derived['TAVR']:.2f}。施設標準（{THRESH['TAVR_low']:.2f}〜{THRESH['TAVR_high']:.2f}）から逸脱。")
    if Re is not None and Re > THRESH["Re_high"]:
        lines.append(f"Re={Re:.0f} と高値で乱流域に近く、渦流・エネルギー損失の関与を考慮。")

    if not lines:
        lines = ["明らかな異常所見なし。経過観察を推奨します。"]
    st.write("・" + "\n・".join(lines))

st.caption("※ 研究用β版。最終判断は臨床背景・画像全体の総合評価で。")
