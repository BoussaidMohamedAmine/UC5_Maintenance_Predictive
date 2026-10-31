"""Interface Streamlit — prédiction de la durée de vie restante (UC5, Jalon 4).

Lancement :
    cd deploy
    .venv/bin/streamlit run streamlit_app.py
"""

from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="UC5 · Prédiction RUL", page_icon="☕", layout="wide")

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

RUL_MAX = 125.0
W = 60
K_AE = 15
CAPTEURS = [f"c{i}" for i in range(8)]
FEATURES = CAPTEURS + ["charge", "HI"]
SEUIL_ALERTE = 30.0

COFFEE = "#6F4E37"
ESPRESSO = "#3E2723"
LATTE = "#A67B5B"
PINK = "#FF8FAB"
PINK_SOFT = "#FFD6E0"
PINK_DEEP = "#E75480"


def _find(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


MODEL_PATH = _find(APP_DIR / "jalon4_final.keras", ROOT / "models" / "jalon4_final.keras")
AE_PATH = _find(ROOT / "models" / "jalon2_autoencodeur.keras", APP_DIR / "jalon2_autoencodeur.keras")
DATA_PATH = _find(ROOT / "data" / "uc5_flotteA.npz", APP_DIR / "data" / "uc5_flotteA.npz")

# --------------------------------------------------------------------------- #
# Style : café + rose frais
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
  background:
    radial-gradient(circle at 10% 0%, #FFE3EC 0%, transparent 45%),
    radial-gradient(circle at 92% 6%, #FFE9DC 0%, transparent 40%),
    linear-gradient(165deg, #FFF9F5 0%, #FFF2F5 55%, #FFE7EE 100%);
}
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1120px; }

h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #3E2723 !important; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #3E2723 0%, #4E342E 55%, #6F4E37 100%);
}
[data-testid="stSidebar"] * { color: #FFF6F0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFD6E0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255,255,255,.09) !important;
  border: 1px solid rgba(255,214,224,.35) !important;
  border-radius: 12px;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,214,224,.22); }

.eyebrow { font-size: .76rem; letter-spacing: .18em; text-transform: uppercase;
           color: #A67B5B; font-weight: 600; }
.lede { color: #6F4E37; font-size: 1rem; margin-top: -.2rem; }

.hero {
  background: linear-gradient(135deg, #FFFFFF 0%, #FFF4F7 58%, #FFE7EE 100%);
  border: 1px solid rgba(231,84,128,.22);
  border-radius: 26px;
  padding: 26px 34px;
  box-shadow: 0 20px 44px -26px rgba(62,39,35,.5),
              0 1px 0 rgba(255,255,255,.9) inset;
}
.hero-label { font-size: .76rem; letter-spacing: .16em; text-transform: uppercase;
              color: #A67B5B; font-weight: 600; }
.hero-value { font-family: 'Fraunces', serif; font-size: 4.8rem; line-height: 1;
              color: #3E2723; font-weight: 700; margin: .3rem 0 .1rem; }
.hero-unit { font-size: 1.1rem; color: #A67B5B; margin-left: .6rem;
             font-family: 'Inter', sans-serif; font-weight: 500; }
.hero-sub { color: #6F4E37; font-size: .95rem; }

.badge { display: inline-block; padding: .26rem .8rem; border-radius: 999px;
         font-size: .76rem; font-weight: 600; letter-spacing: .04em; }
.badge-alerte { background: #E75480; color: #fff; }
.badge-surv { background: #A67B5B; color: #fff; }
.badge-ok { background: #FFD6E0; color: #8C2F4A; }

.tile { background: #fff; border: 1px solid rgba(231,84,128,.16); border-radius: 18px;
        padding: 16px 18px; box-shadow: 0 10px 26px -22px rgba(62,39,35,.5); }
.tile-k { font-size: .72rem; text-transform: uppercase; letter-spacing: .12em;
          color: #A67B5B; font-weight: 600; }
.tile-v { font-family: 'Fraunces', serif; font-size: 1.9rem; color: #3E2723; line-height: 1.2; }
.tile-s { font-size: .78rem; color: #6F4E37; }

[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Modèle + données
# --------------------------------------------------------------------------- #
def _fenetres_ae(df: pd.DataFrame, k: int = K_AE) -> np.ndarray:
    blocs = []
    for _, g in df.groupby("pompe", sort=False):
        v = g[CAPTEURS + ["charge"]].to_numpy(dtype="float32")
        v = np.vstack([np.repeat(v[:1], k - 1, axis=0), v])
        blocs.append(np.stack([v[i : i + k].ravel() for i in range(len(g))]))
    return np.concatenate(blocs)


@st.cache_resource(show_spinner="Chargement du modèle et des données…")
def setup():
    modele = keras.models.load_model(MODEL_PATH, compile=False)
    autoencodeur = keras.models.load_model(AE_PATH, compile=False)

    archive = np.load(DATA_PATH)
    flotte = {}
    for split in ("train", "val", "test"):
        d = pd.DataFrame(archive["capteurs_" + split], columns=CAPTEURS)
        d["pompe"] = archive["pompe_" + split]
        d["cycle"] = archive["cycle_" + split]
        d["charge"] = archive["charge_" + split]
        if "rul_" + split in archive.files:
            d["rul"] = archive["rul_" + split]
        flotte[split] = d.sort_values(["pompe", "cycle"]).reset_index(drop=True)

    train = flotte["train"]
    sains = train[train["rul"] > 150]
    mu = sains.groupby("charge")[CAPTEURS].mean()
    sd = sains.groupby("charge")[CAPTEURS].std()

    tn = train[["pompe", "cycle"] + CAPTEURS].copy()
    tn[CAPTEURS] = (
        train[CAPTEURS].to_numpy() - mu.loc[train["charge"]].to_numpy()
    ) / sd.loc[train["charge"]].to_numpy()
    tn["charge"] = (train["charge"].to_numpy() - 80.0) / 20.0

    fenetres = _fenetres_ae(tn)
    erreur = ((autoencodeur.predict(fenetres, verbose=0) - fenetres) ** 2).mean(axis=1)
    reference = np.log(erreur[train["rul"].to_numpy() > 150] + 1e-8)

    return modele, autoencodeur, flotte, mu, sd, float(reference.mean()), float(reference.std())


def preparer(df: pd.DataFrame, autoencodeur, mu, sd, ref_mean, ref_std) -> pd.DataFrame:
    out = df.sort_values(["pompe", "cycle"]).reset_index(drop=True).copy()
    regimes = np.array(sorted(mu.index))
    charge = regimes[np.abs(out["charge"].to_numpy()[:, None] - regimes[None, :]).argmin(axis=1)]

    out[CAPTEURS] = (
        out[CAPTEURS].to_numpy() - mu.loc[charge].to_numpy()
    ) / sd.loc[charge].to_numpy()
    out["charge"] = (charge - 80.0) / 20.0

    fenetres = _fenetres_ae(out)
    erreur = ((autoencodeur.predict(fenetres, verbose=0) - fenetres) ** 2).mean(axis=1)
    out["HI"] = ((np.log(erreur + 1e-8) - ref_mean) / ref_std).astype("float32")
    return out


def predire(pompe: pd.DataFrame, modele) -> pd.DataFrame:
    valeurs = pompe[FEATURES].to_numpy(dtype="float32")
    derivees = np.vstack(
        [np.zeros((1, valeurs.shape[1]), "float32"), np.diff(valeurs, axis=0)]
    )
    complet = np.concatenate([valeurs, derivees], axis=1)
    rempli = np.vstack([np.zeros((W - 1, complet.shape[1]), "float32"), complet])
    fenetres = sliding_window_view(rempli, W, axis=0).transpose(0, 2, 1)

    res = pd.DataFrame(
        {
            "cycle": pompe["cycle"].to_numpy(),
            "HI": pompe["HI"].to_numpy(),
            "RUL": modele.predict(fenetres, verbose=0).ravel() * RUL_MAX,
        }
    )
    if "rul" in pompe:
        res["RUL réelle"] = pompe["rul"].to_numpy()
    return res


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
if MODEL_PATH is None or AE_PATH is None or DATA_PATH is None:
    st.error(
        "Fichiers manquants. Il faut `deploy/jalon4_final.keras`, "
        "`models/jalon2_autoencodeur.keras` et `data/uc5_flotteA.npz`."
    )
    st.stop()

modele, autoencodeur, flotte, mu, sd, ref_mean, ref_std = setup()

st.markdown('<div class="eyebrow">Étude de cas 5 · Groupe 04 · Jalon 4</div>', unsafe_allow_html=True)
st.title("☕ Durée de vie restante d'une pompe")
st.markdown(
    '<p class="lede">Choisissez une pompe : le modèle LSTM estime sa RUL cycle par cycle.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Réglages")
    split = st.radio(
        "Jeu de données",
        ["Test (historique tronqué)", "Validation (RUL connue)"],
        help="Le test simule une pompe en service : la panne n'est pas encore arrivée.",
    )
    cle = "test" if split.startswith("Test") else "val"

    pompes = sorted(flotte[cle]["pompe"].unique())
    pompe_id = st.selectbox("Pompe", pompes, format_func=lambda p: f"Pompe {p}")

    post = st.toggle(
        "Règle métier (RUL < 45 et HI > 2 → 29)",
        value=False,
        help="Post-traitement du notebook : force l'alerte quand le HI est anormal.",
    )
    st.divider()
    st.caption(f"Modèle LSTM · fenêtre {W} cycles · 20 canaux")
    st.caption(f"{modele.count_params():,} paramètres".replace(",", " "))

brut = flotte[cle][flotte[cle]["pompe"] == pompe_id]
prepare = preparer(brut, autoencodeur, mu, sd, ref_mean, ref_std)
res = predire(prepare, modele)

if post:
    alerte = (res["RUL"] < 45) & (res["HI"] > 2)
    res["RUL"] = res["RUL"].mask(alerte, 29.0)

dernier = res.iloc[-1]
rul = float(dernier["RUL"])
hi = float(dernier["HI"])

if rul < SEUIL_ALERTE:
    badge = '<span class="badge badge-alerte">⚠ Alerte</span>'
elif rul < 2 * SEUIL_ALERTE:
    badge = '<span class="badge badge-surv">Surveillance</span>'
else:
    badge = '<span class="badge badge-ok">Sain</span>'

st.markdown(
    f"""
<div class="hero">
  <div class="hero-label">Durée de vie restante estimée</div>
  <div class="hero-value">{rul:.0f}<span class="hero-unit">cycles</span></div>
  <div class="hero-sub">Pompe {pompe_id} · cycle {int(dernier['cycle'])} · {badge}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="tile"><div class="tile-k">Indicateur de santé</div>'
        f'<div class="tile-v">{hi:.1f}</div>'
        f'<div class="tile-s">0 = sain · &gt; 2 = anormal</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="tile"><div class="tile-k">Cycles observés</div>'
        f'<div class="tile-v">{len(res)}</div>'
        f'<div class="tile-s">du cycle {int(res["cycle"].iloc[0])} au cycle {int(res["cycle"].iloc[-1])}</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    if "RUL réelle" in res:
        vraie = float(dernier["RUL réelle"])
        st.markdown(
            f'<div class="tile"><div class="tile-k">RUL réelle</div>'
            f'<div class="tile-v">{vraie:.0f}</div>'
            f'<div class="tile-s">écart {rul - vraie:+.0f} cycles</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="tile"><div class="tile-k">RUL réelle</div>'
            '<div class="tile-v">—</div>'
            '<div class="tile-s">inconnue sur le jeu de test</div></div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(11.5, 3.9), dpi=150)
fig.patch.set_facecolor("#FFF9F5")
ax.set_facecolor("#FFFFFF")

ax.axhspan(0, SEUIL_ALERTE, color=PINK_SOFT, alpha=0.45, lw=0)
ax.plot(res["cycle"], res["RUL"], color=COFFEE, lw=2.4, label="RUL prédite", zorder=3)
if "RUL réelle" in res:
    ax.plot(res["cycle"], res["RUL réelle"], color=LATTE, lw=1.6, ls="--",
            label="RUL réelle", zorder=2)
ax.axhline(SEUIL_ALERTE, color=PINK_DEEP, lw=1.2, ls=":", label=f"Seuil d'alerte ({SEUIL_ALERTE:.0f})")
ax.set_ylabel("RUL (cycles)", color=COFFEE, fontsize=9)
ax.set_xlabel("Cycle", color=COFFEE, fontsize=9)
ax.tick_params(colors=COFFEE, labelsize=8)
ax.grid(axis="y", color="#F2E4DE", lw=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#E9D8D0")
ax.spines["bottom"].set_color("#E9D8D0")

ax2 = ax.twinx()
ax2.plot(res["cycle"], res["HI"], color=PINK, lw=1.9, alpha=0.9, label="Indicateur de santé (HI)")
ax2.set_ylabel("HI", color=PINK_DEEP, fontsize=9)
ax2.tick_params(colors=PINK_DEEP, labelsize=8)
ax2.spines["top"].set_visible(False)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=8, ncol=4)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Détail
# --------------------------------------------------------------------------- #
with st.expander("Détail des derniers cycles"):
    colonnes = ["cycle", "HI", "RUL"] + (["RUL réelle"] if "RUL réelle" in res else [])
    tableau = res[colonnes].tail(10).iloc[::-1].round(2)
    tableau.columns = ["Cycle", "HI", "RUL prédite", "RUL réelle"][: len(colonnes)]
    st.dataframe(tableau, hide_index=True)

    st.download_button(
        "Télécharger les prédictions (CSV)",
        res[colonnes].round(2).to_csv(index=False).encode("utf-8"),
        file_name=f"pompe_{pompe_id}_rul.csv",
        mime="text/csv",
    )
