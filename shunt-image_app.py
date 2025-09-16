# app.py
# -------------------------------------------------------------
# 透析シャント機能評価「画像読込＋判読サポート」プロトタイプ v0.1
# Streamlit Cloud / ローカルの両対応
# 実行: streamlit run app.py
# -------------------------------------------------------------

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import math

st.set_page_config(page_title="シャント機能評価 画像判読サポート", layout="wide")

# =============== 定数/初期設定 ===============
MAKERS = [
    "GE", "FUJIFILM", "CANON", "日立アロカ", "コニカミノルタ",
    "PHILIPS", "SAMSUNG", "Siemens",
]

# ベンダー別のパラメータ表記 ⇔ 標準化ラベル
ALIASES = {
    "default": {
        "PSV": ["PSV", "Vp", "Vmax", "S"],
        "EDV": ["EDV", "Vd", "D"],
        "TAV": ["TAV", "Vm", "Mean", "Vmean"],
        "TAMV": ["TAMV", "MV", "Time-Avg Mean Velocity", "TAMEAN"],
        "FV": ["Flow", "Q", "V_flow", "VolFlow", "BFV", "FlowVolume"],
        "RI": ["RI", "ResistiveIndex"],
        "PI": ["PI", "PulsatilityIndex"],
    },
    "GE": {
        "PSV": ["PSV", "Vp"],
        "EDV": ["EDV", "Vd"],
        "TAV": ["Vm", "Mean"],
        "TAMV": ["TAMV"],
        "FV": ["Q", "Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
    },
    "FUJIFILM": {
        "PSV": ["PS", "PSV"],
        "EDV": ["ED", "EDV"],
        "TAV": ["Vm", "Mean"],
        "TAMV": ["TAMV"],
        "FV": ["Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
    },
    "CANON": {
        "PSV": ["PSV"],
        "EDV": ["EDV"],
        "TAV": ["TAV"],
        "TAMV": ["TAMV"],
        "FV": ["Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
    },
    "日立アロカ": {
        "PSV": ["Vp", "PSV"],
        "EDV": ["Vd", "EDV"],
        "TAV": ["Vm", "Mean"],
        "TAMV": ["TAMV"],
        "FV": ["Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
    },
    "コニカミノルタ": {
        "PSV": ["PSV"],
        "EDV": ["EDV"],
        "TAV": ["TAV", "Vm"],
        "TAMV": ["TAMV"],
        "FV": ["Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
    },
    "PHILIPS": {
        "PSV": ["PSV"],
        "EDV": ["EDV"],
        "TAV": ["Mean", "Vm"],
        "TAMV": ["TAMV"],
        "FV": ["Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
    },
    "SAMSUNG": {
        "PSV": ["PSV"],
        "EDV": ["EDV"],
        "TAV": ["TAV"],
        "TAMV": ["TAMV"],
        "FV": ["Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
    },
    "Siemens": {
        "PSV": ["PSV"],
        "EDV": ["EDV"],
        "TAV": ["Mean", "Vm"],
        "TAMV": ["TAMV"],
        "FV": ["Flow"],
        "RI": ["RI"],
        "PI": ["PI"],
    },
}

# しきい値（編集可能）
THRESH = {
    "RI_high": 0.68,      # 既知の院内ガイドライン（ユーザー指定）
    "PI_high": 1.25,      # 暫定（データで後で再調整）
    "FV_low": 380.0,      # mL/min
    "TAVR_low": 0.55,
    "TAVR_high": 0.60,
    "SDratio_high": 4.0,  # PSV/EDV 高値目安
    "Re_high": 2300.0,    # 乱流目安
}

# 波形タイプ（I〜V）説明（便宜的プレースホルダ）
WAVE_DESCRIPTIONS = {
    "I":  "高抵抗・三相性に近い。拡張期の逆流あり/低下。末梢抵抗が高い所見。",
    "II": "二相性〜高抵抗寄り。拡張期流速は低め。",
    "III":"単相化し拡張期も一定の順行成分。VA影響で低抵抗化。",
    "IV": "拡張期流速が持続高値。低抵抗パターンが顕著。",
    "V":  "ほぼ等流・平坦化。心周期依存性が乏しく、常時高流指向。",
}

# =============== ユーティリティ ===============
def normalize_labels(maker: str, input_dict: dict) -> dict:
    """メーカーごとの表記ゆれを標準ラベルへ正規化。"""
    aliases = {k: set(v) for k, v in ALIASES.get(maker, ALIASES["default"]).items()}
    std = {"PSV": None, "EDV": None, "TAV": None, "TAMV": None, "FV": None, "RI": None, "PI": None}
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

    # RI/PI 未入力時は PSV/EDV/TAMV から計算
    if RI is None and PSV not in (None, 0):
        RI = (PSV - (EDV if EDV is not None else 0.0)) / PSV
    if PI is None and TAMV not in (None, 0) and PSV is not None and EDV is not None:
        PI = (PSV - EDV) / TAMV

    return {"PSV": PSV, "EDV": EDV, "TAV": TAV, "TAMV": TAMV, "FV": FV, "RI": RI, "PI": PI,
            "TAVR": TAVR, "RIPI": RIPI, "SDratio": SDratio}

def reynolds_number(mean_velocity_m_per_s, vessel_diameter_mm, rho=1060.0, mu=0.0035):
    """Re = rho * v * D / mu"""
    if mean_velocity_m_per_s is None or vessel_diameter_mm is None:
        return None
    D = vessel_diameter_mm / 1000.0
    return (rho * mean_velocity_m_per_s * D) / mu

def suggest_wave_type(ri, pi):
    """簡易ルール（暫定）。後で文献ルールへ置換予定。"""
    if ri is None or pi is None:
        return None
    if ri >= 0.8:      return "I"
    if 0.7 <= ri < 0.8:return "II"
    if 0.6 <= ri < 0.7:return "III"
    if 0.5 <= ri < 0.6:return "IV"
    return "V"

def envelope_from_image(img_bgr, invert=False):
    """グレースケール→Canny→列ごと最上縁→0-1正規化の包絡を返す（簡易版）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    env = np.zeros(w)
    for x in range(w):
        ys = np.where(edges[:, x] > 0)[0]
        if ys.size > 0:
            env[x] = (h - ys.min()) / h
        else:
            env[x] = np.nan
    xs = np.arange(w)
    mask = ~np.isnan(env)
    if mask.sum() >= 2:
        env = np.interp(xs, xs[mask], env[mask])
    else:
        env[:] = 0.0
    return env

def compute_psv_edv_from_envelope(env, px2vel_m_per_s):
    """包絡(0-1)→PSV/EDV（m/s）へ。95%/10%分位を利用（外れ値頑健）。"""
    if env is None or len(env) == 0 or np.nanmax(env) == 0:
        return None, None
    vmax = float(np.nanpercentile(env, 95))
    vmin = float(np.nanpercentile(env, 10))
    PSV = vmax * px2vel_m_per_s
    EDV = vmin * px2vel_m_per_s
    return PSV, EDV

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
maker = st.sidebar.selectbox("メーカーを選択", MAKERS, index=1)
st.sidebar.caption("選択したメーカーに応じてパラメータ名称を自動対応します。")

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

# =============== 画像入力 ===============
st.header("シャント機能評価：画像読込 & 自動解析（β）")

col1, col2 = st.columns([1.2, 1])
with col1:
    uploaded = st.file_uploader("画像をアップロード（JPEG/PNG）", type=["jpg", "jpeg", "png"])
    invert = st.checkbox("明度反転（波形抽出がうまくいかない場合）", value=False)
    st.caption("※ 現状のβ版では簡易的な輪郭抽出で波形包絡を推定します。失敗時は手入力欄をご利用ください。")

    PSV_m_s = None
    EDV_m_s = None

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="入力画像プレビュー", use_column_width=True)
        npimg = np.array(img)[:, :, ::-1]  # PIL(RGB)→OpenCV(BGR)
        env = envelope_from_image(npimg, invert=invert)

        st.subheader("速度スケール校正")
        colv1, colv2 = st.columns(2)
        with colv1:
            vmax_cm_s = st.number_input("画面の最大スケール（上端）[cm/s]", value=150.0, step=10.0)
        with colv2:
            vmin_cm_s = st.number_input("画面の最小スケール（下端）[cm/s]", value=-50.0, step=10.0)

        # 0-1正規化1.0あたりの m/s 換算（単純差分。実画面に合わせて調整可）
        px2vel = (vmax_cm_s - vmin_cm_s) / 100.0
        PSV_m_s, EDV_m_s = compute_psv_edv_from_envelope(env, px2vel)

        st.write("**自動推定（β）**")
        st.metric("PSV [m/s]", f"{PSV_m_s:.2f}" if PSV_m_s is not None else "—")
        st.metric("EDV [m/s]", f"{EDV_m_s:.2f}" if EDV_m_s is not None else "—")
    else:
        st.info("上の枠に JPEG/PNG をドロップするかクリックして選択してください。")

with col2:
    st.subheader("手入力（または値の上書き）")
    st.caption("メーカー表記で入力 → 下で標準ラベルに自動変換します。空欄は自動/未使用。")
    maker_alias = ALIASES.get(maker, ALIASES["default"])

    inputs = {}
    for std_key, alist in maker_alias.items():
        shown = alist[0] if len(alist) else std_key
        val_str = st.text_input(f"{shown}", value="")
        try:
            val = float(val_str) if val_str.strip() != "" else None
        except:
            val = None
        inputs[shown] = val

    # 画像からの自動推定を反映（PSV/EDV）
    if PSV_m_s is not None:
        inputs[maker_alias["PSV"][0]] = PSV_m_s
    if EDV_m_s is not None:
        inputs[maker_alias["EDV"][0]] = EDV_m_s

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
suggested = suggest_wave_type(derived.get("RI"), derived.get("PI"))
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
