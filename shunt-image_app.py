# app.py
# -------------------------------------------------------------
# 透析シャント機能評価 画像読込＋判読サポート v0.7
# 目的: GEの左上パラメータパネルが読めないケースを潰すため、
#       左上の候補範囲を"細かくスライド探索"し、最も多くの
#       ラベル(PS/ED/TAMAX/...)を検出できた窓で二段OCRを実施。
# 追加:
#   - 2.2x アップスケール + バイラテラル/シャープで強調
#   - 緑抽出/白黒の両系統を試行
#   - PSM 11→7→6 の順で再試行
#   - デバッグ: 採用窓/候補窓数/一致数を表示可能
# -------------------------------------------------------------

import streamlit as st
from PIL import Image
import numpy as np
import cv2, re, shutil

# ---- OCR backend ----
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

MAKERS = ["GE", "コニカミノルタ", "FUJIFILM", "CANON", "日立アロカ", "PHILIPS", "SAMSUNG", "Siemens"]

ALIASES = {
    "GE": {
        "PSV": ["PS","PSV"], "EDV": ["ED","EDV"], "TAMV": ["TAMAX","TAMV"],
        "TAV": ["TAMEAN","TAV","MEAN"], "FV": ["FV"], "RI": ["RI"], "PI": ["PI"],
        "VF_Diam": ["VF Diam","VFDiam"],
    },
    "default": {
        "PSV":["PSV","PS"], "EDV":["EDV","ED"], "TAMV":["TAMV","TAMEAN","TAMAX"], "TAV":["TAV","MEAN"],
        "FV":["FV","Flow"], "RI":["RI"], "PI":["PI"], "VF_Diam":["VF Diam","Diameter"],
    },
}

THRESH = {"RI_high":0.68, "PI_high":1.25, "FV_low":380.0, "TAVR_low":0.55, "TAVR_high":0.60,
          "SDratio_high":4.0, "Re_high":2300.0}

WAVE_DESCRIPTIONS = {
    "I":"高抵抗・三相性に近い。拡張期の逆流あり/低下。末梢抵抗が高い所見。",
    "II":"二相性〜高抵抗寄り。拡張期流速は低め。", "III":"単相化し拡張期も一定の順行成分。VA影響で低抵抗化。",
    "IV":"拡張期流速が持続高値。低抵抗パターンが顕著。", "V":"ほぼ等流・平坦化。"
}

# ---------------- OCR補助 ----------------
def _upscale_enhance(rgb):
    # 2.2x アップスケール → バイラテラル → シャープ
    h, w = rgb.shape[:2]
    scale = 2.2
    up = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    up = cv2.bilateralFilter(up, 7, 60, 60)
    # シャープ
    k = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]], dtype=np.float32)
    up = cv2.filter2D(up, -1, k)
    return up

def _prep_bw(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    th = cv2.adaptiveThreshold(eq,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,15)
    return th

def _prep_green(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([30, 30, 30], np.uint8)   # 緑を広めに
    upper = np.array([90,255,255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    g = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(g, 3)
    _, gbin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return gbin

def _tess(img_bin, psm=11, whitelist=None):
    cfg = f"--oem 3 --psm {psm}"
    if whitelist: cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_data(img_bin, lang="eng",
                                         output_type=pytesseract.Output.DICT, config=cfg)
    except Exception:
        return {"text":[], "conf":[], "left":[], "top":[], "width":[], "height":[]}

def _join_lines(data, y_tol=14):
    words=[]
    n=len(data["text"])
    for i in range(n):
        t=(data["text"][i] or "").strip()
        if not t: continue
        words.append((data["top"][i], data["left"][i], t))
    words.sort(key=lambda x:(x[0],x[1]))
    lines=[]; buf=[]; last=None
    for y,x,t in words:
        if last is None or abs(y-last)<=y_tol: buf.append(t)
        else:
            lines.append(" ".join(buf)); buf=[t]
        last=y
    if buf: lines.append(" ".join(buf))
    return lines

# ラベル用パターン
def _label_patterns(maker):
    alias = ALIASES.get(maker, ALIASES["default"])
    pats = {}
    for std, vs in alias.items():
        pats[std] = [re.compile(r"^" + re.escape(v).replace(r"\ ", r"\s*") + r"$", re.IGNORECASE) for v in vs]
    return pats

def _scan_left_top(rgb, maker, debug=False):
    """左上範囲(幅12-30%, 高14-50%)を格子サーチし、最も多くラベル単語が見つかった窓を返す"""
    H, W = rgb.shape[:2]
    x_stops = np.linspace(0.12, 0.30, 7)
    y_stops = np.linspace(0.14, 0.50, 7)
    win_w = int(W*0.16)  # 幅16%
    win_h = int(H*0.20)  # 高20%

    best = None
    best_hits = -1
    best_data = None
    label_pats = _label_patterns(maker)

    for x0r in x_stops:
        for y0r in y_stops:
            x0 = int(W*x0r); y0 = int(H*y0r)
            x1 = min(W, x0+win_w); y1 = min(H, y0+win_h)
            roi = rgb[y0:y1, x0:x1]

            # アップスケール＋2系統前処理
            roi_up = _upscale_enhance(roi)
            mats = [_prep_green(roi_up), _prep_bw(roi_up)]
            hits = 0
            for m in mats:
                d=_tess(m, psm=6)  # 単語
                n=len(d["text"])
                for i in range(n):
                    t=(d["text"][i] or "").strip()
                    if not t: continue
                    for pats in label_pats.values():
                        if any(p.fullmatch(t) for p in pats):
                            hits += 1
            if hits>best_hits:
                best_hits=hits; best=(x0,y0,x1,y1); best_data=(roi_up, mats)

    return best, best_hits, best_data

def _second_pass_numbers(roi_up, label_words=("PS","ED","TAMAX","TAMV","TAMEAN","MEAN","PI","RI","FV","VF","Diam")):
    """
    ROI内で 'PS','ED','TAMAX' 等をまず行で見つけ、
    その行の右側を再OCRして数値を抜く。
    """
    out={}
    # 行検出（緑/白黒のどちらでも）
    mats = [_prep_green(roi_up), _prep_bw(roi_up)]
    lines=[]
    for psm in (11,7,6):
        for m in mats:
            data=_tess(m, psm=psm)
            lines += _join_lines(data, y_tol=16)

    # 正規表現（やや緩い）
    # 例: "PS 93.48 cm/s", "ED: 44.75 cm/s", "TAMAX 58.64 cm/s", "FV 598.79 ml/min", "VF Diam 0.56 cm"
    for ln in lines:
        s = " " + ln + " "
        # GEラベル種（別名も拾う）
        pairs = [
            ("PSV", [r"\bPS\b", r"\bPSV\b"]),
            ("EDV", [r"\bED\b", r"\bEDV\b"]),
            ("TAMV",[r"\bTAMAX\b", r"\bTAMV\b"]),
            ("TAV", [r"\bTAMEAN\b", r"\bTAV\b", r"\bMEAN\b"]),
            ("PI",  [r"\bPI\b"]),
            ("RI",  [r"\bRI\b"]),
            ("FV",  [r"\bFV\b", r"\bFlow\b"]),
            ("VF_Diam", [r"\bVF\s*Diam\b", r"\bVFDiam\b", r"\bDiameter\b"]),
        ]
        for key, labels in pairs:
            for lab in labels:
                m = re.search(lab + r"[^\d\-]{0,6}(-?\d+(?:[.,]\d+)?)", s, re.IGNORECASE)
                if m and out.get(key) is None:
                    try:
                        out[key] = float(m.group(1).replace(",", "."))
                    except:
                        pass
    return out

def ocr_params_ge_auto(pil_img, maker, debug=False):
    if not OCR_AVAILABLE:
        return {k: None for k in ["PSV","EDV","TAV","TAMV","PI","RI","FV","VF_Diam"]}, None, {}

    img = np.array(pil_img.convert("RGB"))
    # 左上範囲を格子探索
    win, hits, data = _scan_left_top(img, maker, debug=debug)
    if win is None:
        return {k: None for k in ["PSV","EDV","TAV","TAMV","PI","RI","FV","VF_Diam"]}, None, {"hits":0}

    x0,y0,x1,y1 = win
    roi = img[y0:y1, x0:x1]
    roi_up = _upscale_enhance(roi)

    vals = _second_pass_numbers(roi_up)
    dbg=None
    if debug:
        dbg = img.copy()
        cv2.rectangle(dbg, (x0,y0), (x1,y1), (0,255,255), 3)
        cv2.putText(dbg, f"hits={hits}", (x0, max(0,y0-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return vals, dbg, {"hits":hits, "win":win}

# ---------------- 解析計算 ----------------
def normalize_labels(maker: str, input_dict: dict) -> dict:
    alias = ALIASES.get(maker, ALIASES["default"])
    std = {"PSV":None,"EDV":None,"TAV":None,"TAMV":None,"FV":None,"RI":None,"PI":None,"VF_Diam":None}
    for k,v in input_dict.items():
        for stdk, names in alias.items():
            if k in names or k==stdk: std[stdk]=v
    return std

def calc_metrics(std: dict) -> dict:
    PSV, EDV, TAV = std.get("PSV"), std.get("EDV"), std.get("TAV")
    TAMV = std.get("TAMV") if std.get("TAMV") is not None else std.get("TAV")
    FV, RI, PI = std.get("FV"), std.get("RI"), std.get("PI")
    TAVR = (TAV/TAMV) if (TAV is not None and TAMV not in (None,0)) else None
    RIPI = (RI/PI) if (RI is not None and PI not in (None,0)) else None
    SDratio = (PSV/max(EDV,1e-9)) if (PSV not in (None,0) and EDV is not None) else None
    if RI is None and PSV not in (None,0): RI=(PSV-(EDV if EDV is not None else 0.0))/PSV
    if PI is None and TAMV not in (None,0) and PSV is not None and EDV is not None: PI=(PSV-EDV)/TAMV
    return {"PSV":PSV,"EDV":EDV,"TAV":TAV,"TAMV":TAMV,"FV":FV,"RI":RI,"PI":PI,"TAVR":TAVR,"RIPI":RIPI,"SDratio":SDratio}

def reynolds_number(v_mean_mps, diam_mm, rho=1060.0, mu=0.0035):
    if v_mean_mps is None or diam_mm is None: return None
    return (rho * v_mean_mps * (diam_mm/1000.0)) / mu

def highlight(val, key):
    if val is None: return "—"
    abnormal=False
    if key=="FV": abnormal = val < THRESH["FV_low"]
    if key=="RI": abnormal = val > THRESH["RI_high"]
    if key=="PI": abnormal = val > THRESH["PI_high"]
    if key=="TAVR": abnormal = (val<THRESH["TAVR_low"]) or (val>THRESH["TAVR_high"])
    if key=="SDratio": abnormal = val > THRESH["SDratio_high"]
    if key=="Re": abnormal = val > THRESH["Re_high"]
    style = "color:red;font-weight:700;" if abnormal else ""
    return f"<span style='{style}'>{val:.3f}" if isinstance(val,float) else f"<span style='{style}'>{val}</span>"

# ---------------- UI ----------------
st.sidebar.header("設定")
maker = st.sidebar.selectbox("メーカーを選択", MAKERS, index=0)
debug_view = st.sidebar.checkbox("デバッグ表示（採用窓）", value=False)

st.sidebar.subheader("血液物性 / 血管径")
rho = st.sidebar.number_input("血液密度 ρ [kg/m³]", value=1060.0, step=10.0, min_value=900.0, max_value=1200.0)
mu  = st.sidebar.number_input("粘度 μ [Pa·s]", value=0.0035, step=0.0001, min_value=0.0010, max_value=0.02)
diam_mm = st.sidebar.number_input("上腕動脈 直径 D [mm]", value=4.0, step=0.5, min_value=1.0, max_value=12.0)

st.sidebar.subheader("基準（赤強調）")
for k in ["RI_high","PI_high","FV_low","TAVR_low","TAVR_high","SDratio_high","Re_high"]:
    THRESH[k] = st.sidebar.number_input(k, value=float(THRESH[k]))

st.header("シャント機能評価：画像読込 & 自動解析（v0.7 / GE左上マルチサーチ）")

col1, col2 = st.columns([1.2,1])
with col1:
    up = st.file_uploader("画像をアップロード（JPEG/PNG）", type=["jpg","jpeg","png"])
    pil_img=None; ocr_std=None; dbg=None; meta={}
    if up is not None:
        pil_img = Image.open(up).convert("RGB")
        st.image(pil_img, caption="入力画像プレビュー", use_container_width=True)

        if OCR_AVAILABLE and maker in ["GE","コニカミノルタ"]:
            with st.spinner("OCR抽出中...（左上スライディング探索→二段OCR）"):
                ocr_std, dbg, meta = ocr_params_ge_auto(pil_img, maker, debug=debug_view)
            if any(v is not None for v in ocr_std.values()):
                st.success(f"OCR抽出（標準キー）/ hits={meta.get('hits',0)}")
                st.json(ocr_std)
            else:
                st.info("OCRで有効なパラメータが抽出できませんでした。")
        elif not OCR_AVAILABLE:
            st.warning("Tesseract 未インストールのためOCRは無効です。")
        else:
            st.info("このメーカーは現状、手入力主体です。")

        if debug_view and dbg is not None:
            st.image(dbg, caption=f"採用された窓（hits={meta.get('hits',0)}）", use_container_width=True)

with col2:
    st.subheader("手入力（または OCR 値の上書き）")
    alias = ALIASES.get(maker, ALIASES["default"])
    inputs={}
    # OCR結果→代表表記へ
    if ocr_std:
        for std_key, vs in alias.items():
            rep = vs[0] if vs else std_key
            inputs[rep] = ocr_std.get(std_key)
    else:
        for std_key, vs in alias.items():
            rep = vs[0] if vs else std_key
            inputs[rep] = None
    for show, initv in inputs.items():
        s = "" if initv is None else str(initv)
        v = st.text_input(show, value=s)
        try: inputs[show] = float(v) if v.strip()!="" else None
        except: inputs[show]=None

# ---- 計算/表示 ----
std = normalize_labels(maker, inputs)
derived = calc_metrics(std)
mean_vel = derived.get("TAV") if derived.get("TAV") not in (None,0) else derived.get("TAMV")
Re = reynolds_number(mean_vel, diam_mm, rho=rho, mu=mu)

st.subheader("標準化パラメータ & 自動算出")
cols = st.columns(4)
for i,k in enumerate(["PSV","EDV","TAV","TAMV"]):
    with cols[i%4]: st.markdown(f"**{k}**: {highlight(derived.get(k),k)}", unsafe_allow_html=True)
cols = st.columns(4)
for i,k in enumerate(["FV","RI","PI","TAVR"]):
    with cols[i%4]: st.markdown(f"**{k}**: {highlight(derived.get(k),k)}", unsafe_allow_html=True)
cols = st.columns(3)
for i,k in enumerate(["RIPI","SDratio"]):
    with cols[i%3]: st.markdown(f"**{k}**: {highlight(derived.get(k),k)}", unsafe_allow_html=True)
st.markdown(f"**Re（Reynolds数）**: {highlight(Re,'Re')}", unsafe_allow_html=True)

def suggest_wave_type(ri, pi):
    if ri is None or pi is None: return None
    if ri>=0.8: return "I"
    if 0.7<=ri<0.8: return "II"
    if 0.6<=ri<0.7: return "III"
    if 0.5<=ri<0.6: return "IV"
    return "V"
suggested = suggest_wave_type(derived.get("RI"), derived.get("PI"))
c1,c2 = st.columns([1,2])
with c1:
    wave = st.selectbox("上腕動脈波形タイプ（自動推定/編集可）",
        options=["—","I","II","III","IV","V"],
        index=(["—","I","II","III","IV","V"].index(suggested) if suggested in ["I","II","III","IV","V"] else 0))
with c2:
    st.info(WAVE_DESCRIPTIONS.get(wave,"現状はRI/PIの暫定境界で推定。"))

with st.expander("最終コメント（雛形）"):
    L=[]
    if derived.get("RI") and derived["RI"]>THRESH["RI_high"]: L.append(f"RI高値（{derived['RI']:.2f}）")
    if derived.get("PI") and derived["PI"]>THRESH["PI_high"]: L.append(f"PI高値（{derived['PI']:.2f}）")
    if derived.get("FV") and derived["FV"]<THRESH["FV_low"]: L.append(f"FV低値（{derived['FV']:.0f} mL/min）")
    if derived.get("TAVR") and (derived['TAVR']<THRESH['TAVR_low'] or derived['TAVR']>THRESH['TAVR_high']): L.append(f"TAVR逸脱（{derived['TAVR']:.2f}）")
    if Re and Re>THRESH["Re_high"]: L.append(f"Re={Re:.0f} 高値（乱流域近傍）")
    if not L: L=["明らかな異常所見なし。経過観察。"]
    st.write("・" + "\n・".join(L))

st.caption("※ v0.7: GE左上パネルをグリッド探索して抽出。読み取りが不安定なら画像原寸に近い解像度でアップロードしてください。")
