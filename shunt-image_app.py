# app.py
# -------------------------------------------------------------
# 透析シャント機能評価「画像読込＋判読サポート」v0.5 (GE強化)
# 変更点:
#  - 左半分で緑(+)クラスタを検出→カード枠ROIを自動推定（位置固定に依存しない）
#  - ROIに対して 3系統前処理（緑抽出/白黒/反転）× PSM(11→7→6) で総当たりOCR
#  - OCR後の文字修正（0/O, 1/l, cm/s 誤認 等）と正規表現のゆるめマッチ
#  - デバッグ表示: ROIとマスクのオーバーレイを確認できる
# -------------------------------------------------------------

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import re
import math
import shutil

# ---- OCR backend（pytesseract）存在チェック ----
OCR_AVAILABLE = True
try:
    import pytesseract
    _tess = shutil.which("tesseract")
    if _tess:
        pytesseract.pytesseract.tesseract_cmd = _tess
    else:
        OCR_AVAILABLE = False
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="シャント機能評価 画像判読サポート", layout="wide")

# =============== 定数/初期設定 ===============
MAKERS = [
    "GE", "コニカミノルタ", "FUJIFILM", "CANON",
    "日立アロカ", "PHILIPS", "SAMSUNG", "Siemens",
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
    "コニカミノルタ": {
        "PSV": ["PSV", "PS"],
        "EDV": ["EDV", "ED"],
        "TAMV": ["TAMV", "TAMAX", "Vm"],
        "TAV": ["TAV", "MEAN"],
        "FV": ["FV", "Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
        "VF_Diam": ["VF Diam", "Diameter"],
    },
}

# メーカー別OCR 検索基準（左半分を優先探索）
ROI_HINT = {
    "GE":            {"x": (0.0, 0.55), "y": (0.0, 0.65)},  # 左上〜中央に多い
    "コニカミノルタ": {"x": (0.0, 0.55), "y": (0.0, 1.00)},
    "default":       {"x": (0.0, 1.00), "y": (0.0, 1.00)},
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

# ========= OCR: GEカード自動検出（緑+クラスタ → 矩形推定） =========
def _candidate_panel_roi(rgb: np.ndarray, maker: str):
    """左半分（メーカーごとのヒント領域）で緑(+)を探し、最も密な縦長ブロブをカード候補として返す"""
    h, w, _ = rgb.shape
    hint = ROI_HINT.get(maker, ROI_HINT["default"])
    x0, x1 = int(w*hint["x"][0]), int(w*hint["x"][1])
    y0, y1 = int(h*hint["y"][0]), int(h*hint["y"][1])
    crop = rgb[y0:y1, x0:x1]

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    lower = np.array([35, 40, 40], np.uint8)
    upper = np.array([85, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # 緑 “+” を繋げる
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_d = cv2.dilate(cv2.erode(mask, k, 1), k, 2)

    cnts, _ = cv2.findContours(mask_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = None
    best_score = -1
    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        area = wc*hc
        if area < 200:  # 小さすぎるノイズは除外
            continue
        # 縦長で左寄りのクラスタを優先
        aspect = hc / max(wc,1)
        leftness = 1.0 - (x / max((x1-x0),1))
        score = aspect*0.6 + leftness*0.4 + np.sqrt(area)/300.0
        if score > best_score:
            best_score = score
            cand = (x,y,wc,hc)

    if cand is None:
        # 見つからなければヒント領域全体を返す
        return (x0, y0, x1-x0, y1-y0), mask_d, crop

    # “+”クラスタからカード全体へ拡張（右と下に広げる）
    x,y,wc,hc = cand
    pad = 20
    X0 = max(x0 + x - pad, 0)
    Y0 = max(y0 + y - pad, 0)
    X1 = min(x0 + x + wc + 260, w)     # 右へ大きめに伸ばす（数字/単位を含める）
    Y1 = min(y0 + y + hc + 220, h)     # 下へ拡張
    return (X0, Y0, X1-X0, Y1-Y0), mask_d, crop

def _prep_variants(roi_rgb: np.ndarray):
    """OCR前処理のバリエーションを返す（バイナリ画像群）"""
    outs = []

    # 緑抽出
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([35, 40, 40], np.uint8)
    upper = np.array([85, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    g = cv2.bitwise_and(roi_rgb, roi_rgb, mask=mask)
    g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(g, 3)
    _, gbin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    outs.append(("green", gbin))

    # 白黒 (CLAHE + 自適応二値)
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    bw = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    outs.append(("bw", bw))

    # 反転（白文字っぽいときの保険）
    outs.append(("inv", 255-bw))

    return outs

def _tess_lines(img_bin: np.ndarray, psm: int, whitelist: str = None):
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(img_bin, lang="eng",
                                         output_type=pytesseract.Output.DICT,
                                         config=cfg)
    except Exception:
        return 0, []

    words, confs = [], []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except Exception:
            conf = -1
        x, y = data["left"][i], data["top"][i]
        words.append((y, x, txt))
        confs.append(conf)

    if not words:
        return 0, []

    words.sort(key=lambda t: (t[0], t[1]))
    lines, buf, last_y = [], [], None
    for y, x, txt in words:
        if last_y is None or abs(y - last_y) <= 14:
            buf.append(txt)
        else:
            lines.append(" ".join(buf))
            buf = [txt]
        last_y = y
    if buf:
        lines.append(" ".join(buf))

    valid = [c for c in confs if c >= 0]
    avg_conf = int(sum(valid)/len(valid)) if valid else 0
    return avg_conf, lines

def _postfix_text_fix(s: str):
    # よくある誤認を補正（O↔0, l↔1, ‘cm/s’の崩れ等）
    s = s.replace("O.", "0.").replace("O,", "0,").replace("O", "0")
    s = s.replace("l.", "1.").replace("l,", "1,").replace("l", "1")
    s = re.sub(r"c[mrn]/[s5]", "cm/s", s, flags=re.IGNORECASE)  # cm/s の崩れ
    s = s.replace("ml/rnin", "ml/min").replace("m1/min", "ml/min")
    s = s.replace("cm5", "cm/s").replace("cmis", "cm/s").replace("cmls", "cm/s")
    s = s.replace("VFD iam", "VF Diam").replace("VFDlam", "VF Diam")
    return s

def _ocr_card(rgb: np.ndarray):
    """カードROIに対して総当たりで行テキストを返す"""
    mats = _prep_variants(rgb)
    collected = []
    for name, m in mats:
        for psm in (11, 7, 6):
            conf, lines = _tess_lines(m, psm=psm, whitelist="+.:-/%cmnilsFVAPTEDRI0123456789")
            fixed = [_postfix_text_fix(x) for x in lines]
            collected.extend(fixed)
            # ある程度読めていれば打ち切り（ヒューリスティック）
            if conf >= 60 and any(("cm/s" in x or "ml/min" in x) for x in fixed):
                return collected
    return collected

def _extract_ge_anywhere(pil_img: Image.Image, maker: str, debug=False):
    img = np.array(pil_img.convert("RGB"))
    (X, Y, W, H), mask_debug, hint_crop = _candidate_panel_roi(img, maker)
    X2, Y2 = X+W, Y+H
    X, Y = max(0,X), max(0,Y)
    X2, Y2 = min(img.shape[1],X2), min(img.shape[0],Y2)
    roi = img[Y:Y2, X:X2].copy()

    lines = _ocr_card(roi)

    # デバッグ画像
    dbg = None
    if debug:
        dbg = img.copy()
        cv2.rectangle(dbg, (X,Y), (X2,Y2), (0,255,255), 3)
        # 左ヒント窓にも表示
        h,w,_ = img.shape
        hx0, hx1 = int(w*ROI_HINT[maker]["x"][0]), int(w*ROI_HINT[maker]["x"][1])
        hy0, hy1 = int(h*ROI_HINT[maker]["y"][0]), int(h*ROI_HINT[maker]["y"][1])
        cv2.rectangle(dbg, (hx0,hy0), (hx1,hy1), (255,128,0), 2)

    return roi, lines, dbg

# 値の抽出（GE/コニカミノルタ共通のゆるめ正規表現）
_VALUE = r"(-?\d+(?:[.,]\d+)?)"
def _parse_params_from_lines(lines, maker):
    pats = {
        "PSV":     [r"\bPSV\b", r"\bPS\b"],
        "EDV":     [r"\bEDV\b", r"\bED\b"],
        "TAMV":    [r"\bTAMV\b", r"\bTAMAX\b"],
        "TAV":     [r"\bTAV\b", r"\bTAMEAN\b", r"\bMEAN\b"],
        "PI":      [r"\bPI\b"],
        "RI":      [r"\bRI\b"],
        "FV":      [r"\bFV\b", r"\bFlow\b"],
        "VF_Diam": [r"\bVF\s*Diam\b", r"\bDiameter\b", r"\bVFDiam\b"],
    }
    out = {k: None for k in pats.keys()}

    for ln in lines:
        s = " " + ln + " "
        for key, alias_list in pats.items():
            for alias in alias_list:
                m = re.search(alias + r"\s*[:=]?\s*" + _VALUE, s, re.IGNORECASE)
                if not m:
                    # キーと値の間に単位が挟まっても許容
                    m = re.search(alias + r".{0,6}" + _VALUE + r"\s*(cm/s|ml/min|cm)?", s, re.IGNORECASE)
                if m and out[key] is None:
                    v = m.group(1).replace(",", ".")
                    try:
                        out[key] = float(v)
                    except:
                        pass
    return out

def extract_params_anywhere(pil_img: Image.Image, maker: str, debug=False):
    if not OCR_AVAILABLE:
        return {k: None for k in ["PSV","EDV","TAV","TAMV","PI","RI","FV","VF_Diam"]}, [], None
    roi, lines, dbg = _extract_ge_anywhere(pil_img, maker, debug=debug)
    params = _parse_params_from_lines(lines, maker)
    return params, lines, (roi, dbg)

# =============== 計算系は従来どおり ===============
def normalize_labels(maker: str, input_dict: dict) -> dict:
    aliases = {k: set(v) for k, v in ALIASES.get(maker, ALIASES["default"]).items()}
    std = {"PSV": None, "EDV": None, "TAV": None, "TAMV": None, "FV": None, "RI": None, "PI": None, "VF_Diam": None}
    for key, val in input_dict.items():
        for stdk, names in aliases.items():
            if key in names or key == stdk:
                std[stdk] = val
    return std

def calc_metrics(std: dict) -> dict:
    PSV = std.get("PSV"); EDV = std.get("EDV")
    TAV = std.get("TAV");  TAMV = std.get("TAMV") if std.get("TAMV") is not None else TAV
    FV  = std.get("FV");   RI  = std.get("RI");   PI  = std.get("PI")

    TAVR = RIPI = SDratio = None
    if TAV is not None and TAMV not in (None, 0): TAVR = TAV / TAMV
    if RI is not None and PI not in (None, 0):    RIPI = RI / PI
    if PSV not in (None, 0) and EDV is not None:  SDratio = PSV / max(EDV, 1e-9)

    if RI is None and PSV not in (None, 0):       RI = (PSV - (EDV if EDV is not None else 0.0)) / PSV
    if PI is None and TAMV not in (None, 0) and PSV is not None and EDV is not None:
        PI = (PSV - EDV) / TAMV

    return {"PSV": PSV, "EDV": EDV, "TAV": TAV, "TAMV": TAMV, "FV": FV, "RI": RI, "PI": PI,
            "TAVR": TAVR, "RIPI": RIPI, "SDratio": SDratio}

def reynolds_number(mean_velocity_m_per_s, vessel_diameter_mm, rho=1060.0, mu=0.0035):
    if mean_velocity_m_per_s is None or vessel_diameter_mm is None:
        return None
    D = vessel_diameter_mm / 1000.0
    return (rho * mean_velocity_m_per_s * D) / mu

def highlight(val, key):
    if val is None: return "—"
    abnormal = False
    if key == "FV" and val is not None:       abnormal = val < THRESH["FV_low"]
    if key == "RI" and val is not None:       abnormal = val > THRESH["RI_high"]
    if key == "PI" and val is not None:       abnormal = val > THRESH["PI_high"]
    if key == "TAVR" and val is not None:     abnormal = (val < THRESH["TAVR_low"]) or (val > THRESH["TAVR_high"])
    if key == "SDratio" and val is not None:  abnormal = val > THRESH["SDratio_high"]
    if key == "Re" and val is not None:       abnormal = val > THRESH["Re_high"]
    style = "color: red; font-weight: 700;" if abnormal else ""
    txt = f"{val:.3f}" if isinstance(val, float) else f"{val}"
    return f"<span style='{style}'>{txt}</span>"

# =============== UI ===============
st.sidebar.header("設定")
maker = st.sidebar.selectbox("メーカーを選択", MAKERS, index=0)
debug_view = st.sidebar.checkbox("デバッグ表示（ROI/マスク）", value=False)

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

st.header("シャント機能評価：画像読込 & 自動解析（v0.5 / GE強化）")

col1, col2 = st.columns([1.2, 1])
with col1:
    uploaded = st.file_uploader("画像をアップロード（JPEG/PNG）", type=["jpg","jpeg","png"])
    if not OCR_AVAILABLE:
        st.warning("OCR(Tesseract)が未インストールのため自動抽出が無効です。"
                   "Streamlit Cloudではリポジトリ直下の packages.txt に "
                   "`tesseract-ocr` と `tesseract-ocr-eng` を記載してください。")

    pil_img = None
    ocr_std = None
    dbg_imgs = None

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="入力画像プレビュー", use_container_width=True)

        if OCR_AVAILABLE:
            with st.spinner("OCR抽出中...（緑+クラスタ→カード切り出し）"):
                ocr_std, ocr_lines, debug_bundle = extract_params_anywhere(pil_img, maker, debug=debug_view)
                dbg_imgs = debug_bundle
            if any(v is not None for v in ocr_std.values()):
                st.success("OCR抽出（標準キー）")
                st.json(ocr_std)
            else:
                st.info("OCRで有効なパラメータが抽出できませんでした。手入力欄をご利用ください。")

        if debug_view and dbg_imgs is not None:
            roi_rgb, dbg = dbg_imgs
            st.caption("推定カードROI（黄色）／ヒント領域（オレンジ）")
            if dbg is not None:
                st.image(dbg, use_container_width=True, caption="Debug Overlay")
            st.image(roi_rgb, use_container_width=True, caption="ROI crop for OCR")

with col2:
    st.subheader("手入力（または OCR 値の上書き）")
    maker_alias = ALIASES.get(maker, ALIASES["default"])
    inputs = {}
    for std_key, alist in maker_alias.items():
        show_name = alist[0] if len(alist) else std_key
        inputs[show_name] = None

    if OCR_AVAILABLE and uploaded is not None and ocr_std:
        for std_key, val in ocr_std.items():
            if val is None: continue
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

# ---- 計算 & 表示 ----
std      = normalize_labels(maker, inputs)
derived  = calc_metrics(std)
mean_vel = derived.get("TAV") if derived.get("TAV") not in (None, 0) else derived.get("TAMV")
Re       = reynolds_number(mean_vel, vessel_diameter_mm=diam_mm, rho=rho, mu=mu)

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

def suggest_wave_type(ri, pi):
    if ri is None or pi is None: return None
    if ri >= 0.8:      return "I"
    if 0.7 <= ri < 0.8:return "II"
    if 0.6 <= ri < 0.7:return "III"
    if 0.5 <= ri < 0.6:return "IV"
    return "V"
suggested = suggest_wave_type(derived.get("RI"), derived.get("PI"))
colw1, colw2 = st.columns([1,2])
with colw1:
    wave_type = st.selectbox("自動推定（編集可）",
        options=["—","I","II","III","IV","V"],
        index=(["—","I","II","III","IV","V"].index(suggested) if suggested in ["I","II","III","IV","V"] else 0)
    )
with colw2:
    st.info(WAVE_DESCRIPTIONS.get(wave_type, "現状はRI/PIの暫定境界で推定。"))

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
