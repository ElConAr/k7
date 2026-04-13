# =============================================================================
# PROYECTO FINAL — Equipo K7  |  Sep–Nov 2025
# Python para Analista de Datos en la Industria Petrolera
#
# Ejecutar en Spyder con F5
# Los gráficos se despliegan en ventanas interactivas dentro de Spyder
#
# Módulos:
#   1. Ingesta, limpieza y estadística descriptiva
#   2. Modelo predictivo GRU con TensorFlow / Keras
#   3. Detección de anomalías Z-Score (con señales precursoras)
#   4. Exportación a PDF con ReportLab
#
# Instalación previa (ejecutar una sola vez en la consola de Spyder):
#   pip install tensorflow pandas numpy matplotlib scipy scikit-learn reportlab
# =============================================================================

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silencia logs de TF
warnings.filterwarnings("ignore")

# ── Rutas del proyecto ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "K7_SepOctNov.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# ── Matplotlib — ventanas interactivas en Spyder ─────────────────
import matplotlib
matplotlib.use("Qt5Agg")          # ventanas flotantes en Spyder
# Si prefieres inline: comenta la línea anterior y descomenta:
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter

plt.rcParams.update({
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#8b949e",
    "text.color"       : "#c9d1d9",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "grid.color"       : "#21262d",
    "grid.linewidth"   : 0.5,
    "lines.linewidth"  : 1.0,
    "font.size"        : 9,
})

# ── Librerías ────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

print(f"[✓] TensorFlow {tf.__version__}  |  "
      f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ── Semillas para reproducibilidad ───────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Paleta de colores ────────────────────────────────────────────
BG, PANEL, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT   = "#c9d1d9"
ACCENT = "#58a6ff"
RED    = "#ff4444"
ORANGE = "#e67e22"
GREEN  = "#2ecc71"
YELLOW = "#f1c40f"
PURPLE = "#9b59b6"

EV_COLORS = {
    "14/SEP": "#e74c3c",
    "20/OCT": "#e67e22",
    "31/OCT": "#9b59b6",
    "11/NOV": "#1abc9c",
}

FMT_DATE = DateFormatter("%d-%b")
FMT_HOUR = DateFormatter("%H:%M")

# =============================================================================
# MÓDULO 1 — INGESTA, LIMPIEZA Y ESTADÍSTICA
# =============================================================================
print("\n" + "="*65)
print("  MÓDULO 1: INGESTA, LIMPIEZA Y ESTADÍSTICA")
print("="*65)

# ── 1.1 Carga ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, sep=";", decimal=",",
                 engine="python", on_bad_lines="skip")

# ── 1.2 Limpieza ─────────────────────────────────────────────────
df.columns     = [c.strip().replace(" ", "_") for c in df.columns]
df.drop(columns=[df.columns[14]], inplace=True)            # columna vacía
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"],
                                  dayfirst=True, errors="coerce")
df.sort_values("TIMESTAMP", inplace=True)
df.reset_index(drop=True, inplace=True)

VARIABLES    = [c for c in df.columns if c != "TIMESTAMP"]
nulos_antes  = df[VARIABLES].isnull().sum().sum()
df[VARIABLES] = df[VARIABLES].interpolate("linear").ffill().bfill()

print(f"[✓] Registros       : {len(df):,}")
print(f"[✓] Variables       : {VARIABLES}")
print(f"[✓] Nulos imputados : {nulos_antes} → 0")
print(f"[✓] Rango temporal  : {df['TIMESTAMP'].min()} → {df['TIMESTAMP'].max()}")

# ── 1.3 Momentos de parada ───────────────────────────────────────
FECHAS = ["2025-09-14", "2025-10-20", "2025-10-31", "2025-11-11"]
LABELS = ["14/SEP", "20/OCT", "31/OCT", "11/NOV"]

SHUTDOWNS = {}
print("\n--- Paradas detectadas (caída brusca de RPM) ---")
for fp, label in zip(FECHAS, LABELS):
    t0   = pd.Timestamp(fp)
    mask = (df["TIMESTAMP"] >= t0) & (df["TIMESTAMP"] < t0 + pd.Timedelta(hours=24))
    sub  = df[mask]
    idx  = sub["RPM"].diff().idxmin()
    t_stop = sub.loc[idx, "TIMESTAMP"]
    SHUTDOWNS[label] = t_stop
    print(f"  [{label}]  {t_stop.strftime('%Y-%m-%d %H:%M')}  "
          f"ΔRPM = {sub['RPM'].diff()[idx]:.0f}")

# ── 1.4 Estadística descriptiva ───────────────────────────────────
print("\n--- Estadística Descriptiva ---")
desc = df[VARIABLES].describe().round(3)
print(desc.to_string())

# ── 1.5 Gráfico 1: Series temporales ─────────────────────────────
COLS_G1  = ["RPM", "PSUCC", "PDESC", "TOIL_ENG", "POIL_ENG"]
COLS_CLR = [ACCENT, YELLOW, "#3498db", ORANGE, GREEN]

fig1, axes = plt.subplots(len(COLS_G1), 1, figsize=(16, 12),
                           sharex=True, num="G1 — Series Temporales")
fig1.suptitle("K7 — Series Temporales | Sep–Nov 2025",
              fontsize=13, fontweight="bold")
for ax, col, c in zip(axes, COLS_G1, COLS_CLR):
    ax.plot(df["TIMESTAMP"], df[col], color=c, lw=0.5, alpha=0.8)
    for lbl, t_stop in SHUTDOWNS.items():
        ax.axvline(t_stop, color=EV_COLORS[lbl], lw=1.3,
                   ls="--", alpha=0.9, label=lbl)
    ax.set_ylabel(col, fontsize=8)
    ax.grid(True)
axes[0].legend(loc="upper right", fontsize=7, ncol=4,
               facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
axes[-1].xaxis.set_major_formatter(FMT_DATE)
plt.tight_layout()
fig1.savefig(os.path.join(OUTPUTS_DIR, "G1_series_temporales.png"),
             dpi=150, bbox_inches="tight", facecolor=BG)
print("\n[✓] G1_series_temporales.png guardado")
plt.show(block=False)

# =============================================================================
# MÓDULO 2 — MODELO GRU (TensorFlow / Keras)
# =============================================================================
print("\n" + "="*65)
print("  MÓDULO 2: MODELO PREDICTIVO GRU — TensorFlow/Keras")
print("="*65)

# ── 2.1 Configuración ────────────────────────────────────────────
TARGET_VARS  = ["TOIL_ENG", "POIL_ENG"]   # multivariado
SEQ_LEN      = 60      # ventana entrada (60 min)
PRED_LEN     = 10      # horizonte predicción (10 min)
BATCH_SIZE   = 64
EPOCHS       = 80
LR           = 0.001
N_VARS       = len(TARGET_VARS)

# ── 2.2 Secuencias de entrenamiento (baseline Sep 1–9) ────────────
baseline_mask = df["TIMESTAMP"] < "2025-09-10"
df_base       = df[baseline_mask].copy()

scaler       = MinMaxScaler()
base_scaled  = scaler.fit_transform(df_base[TARGET_VARS])
all_scaled   = scaler.transform(df[TARGET_VARS])

def make_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = make_sequences(base_scaled, SEQ_LEN, PRED_LEN)
X_all,   y_all   = make_sequences(all_scaled,  SEQ_LEN, PRED_LEN)

y_train_flat = y_train.reshape(len(y_train), PRED_LEN * N_VARS)
y_all_flat   = y_all.reshape(len(y_all),     PRED_LEN * N_VARS)

print(f"[✓] X_train : {X_train.shape}   y_train : {y_train_flat.shape}")
print(f"[✓] X_all   : {X_all.shape}     y_all   : {y_all_flat.shape}")

# ── 2.3 Arquitectura GRU ─────────────────────────────────────────
#   GRU(64) → Dropout → GRU(32) → Dropout → Dense(64,relu) → Dense(output)
def build_gru(seq_len, n_vars, pred_len):
    inp = keras.Input(shape=(seq_len, n_vars), name="input")
    x   = layers.GRU(64, return_sequences=True,
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="gru_1")(inp)
    x   = layers.Dropout(0.2, name="drop_1")(x)
    x   = layers.GRU(32, return_sequences=False,
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="gru_2")(x)
    x   = layers.Dropout(0.2, name="drop_2")(x)
    x   = layers.Dense(64, activation="relu", name="dense_1")(x)
    out = layers.Dense(pred_len * n_vars, activation="linear",
                       name="output")(x)
    model = keras.Model(inputs=inp, outputs=out, name="GRU_K7")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
                  loss="mse", metrics=["mae"])
    return model

model = build_gru(SEQ_LEN, N_VARS, PRED_LEN)
model.summary()

# ── 2.4 Callbacks ────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODEL_DIR, "gru_k7_best.keras")
cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=10,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=5, min_lr=1e-5, verbose=1),
    callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss",
                              save_best_only=True, verbose=0),
]

# ── 2.5 Entrenamiento ────────────────────────────────────────────
print(f"\n[→] Entrenando GRU (máx {EPOCHS} épocas, "
      f"batch={BATCH_SIZE}, lr={LR})...")
history = model.fit(
    X_train, y_train_flat,
    epochs           = EPOCHS,
    batch_size       = BATCH_SIZE,
    validation_split = 0.15,
    callbacks        = cb_list,
    shuffle          = True,
    verbose          = 1,
)
ep_real = len(history.history["loss"])
print(f"[✓] Entrenamiento completado ({ep_real} épocas efectivas)")
print(f"    Mejor val_loss : {min(history.history['val_loss']):.6f}")

# ── 2.6 Error de reconstrucción ───────────────────────────────────
print("[→] Calculando errores de reconstrucción...")
y_pred_flat = model.predict(X_all, batch_size=256, verbose=0)
recon_errors = np.mean((y_pred_flat - y_all_flat) ** 2, axis=1)

ts_pred    = df["TIMESTAMP"].iloc[SEQ_LEN : SEQ_LEN + len(recon_errors)].values
ts_pred_dt = pd.to_datetime(ts_pred)

# Métricas desnormalizadas
y_real_dn = scaler.inverse_transform(
    y_all_flat.reshape(-1, N_VARS)).flatten()
y_pred_dn = scaler.inverse_transform(
    y_pred_flat.reshape(-1, N_VARS)).flatten()
mae_val  = mean_absolute_error(y_real_dn, y_pred_dn)
rmse_val = np.sqrt(mean_squared_error(y_real_dn, y_pred_dn))

# Umbral: media + 3σ del error en baseline
n_base_seq  = len(X_train)
err_base    = recon_errors[:n_base_seq]
thr_gru     = err_base.mean() + 3 * err_base.std()
gru_anomaly = recon_errors > thr_gru

print(f"[✓] MAE = {mae_val:.4f}  |  RMSE = {rmse_val:.4f}")
print(f"[✓] Umbral GRU (μ+3σ) = {thr_gru:.6f}  |  "
      f"Anomalías detectadas: {gru_anomaly.sum():,}")

# ── 2.7 Gráfico 2: Curvas de entrenamiento ────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 4),
                           num="G2 — Curvas GRU")
fig2.suptitle("GRU (TF/Keras) — Curvas de Entrenamiento",
              fontsize=12, fontweight="bold")
for ax, metric, c_tr, c_vl, ylabel in [
    (axes[0], "loss",   ACCENT, ORANGE, "MSE"),
    (axes[1], "mae",    GREEN,  RED,    "MAE"),
]:
    ax.plot(history.history[metric],
            color=c_tr, lw=1.5, label=f"Train {ylabel}")
    ax.plot(history.history[f"val_{metric}"],
            color=c_vl, lw=1.5, label=f"Val {ylabel}")
    ax.fill_between(range(ep_real),
                    history.history[metric], color=c_tr, alpha=0.10)
    ax.set_xlabel("Época"); ax.set_ylabel(ylabel); ax.grid(True)
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUTS_DIR, "G2_curvas_gru.png"),
             dpi=150, bbox_inches="tight", facecolor=BG)
print("[✓] G2_curvas_gru.png guardado")
plt.show(block=False)

# ── 2.8 Gráfico 3: Error reconstrucción serie completa ───────────
fig3, ax = plt.subplots(figsize=(16, 5), num="G3 — Error GRU")
fig3.suptitle(f"GRU — Error de Reconstrucción (MSE) | "
              f"Umbral μ+3σ = {thr_gru:.5f}",
              fontsize=11, fontweight="bold")
ax.fill_between(ts_pred_dt, recon_errors, color=RED, alpha=0.25,
                label="Error reconstrucción")
ax.fill_between(ts_pred_dt, recon_errors,
                where=(recon_errors > thr_gru),
                color=RED, alpha=0.70, label="Anomalía GRU")
ax.plot(ts_pred_dt, recon_errors, color=RED, lw=0.4, alpha=0.5)
ax.axhline(thr_gru, color=YELLOW, lw=1.2, ls="--",
           label=f"Umbral = {thr_gru:.5f}")
for lbl, t_stop in SHUTDOWNS.items():
    ax.axvline(t_stop, color=EV_COLORS[lbl], lw=1.5,
               ls="--", alpha=0.9, label=f"Parada {lbl}")
ax.set_ylabel("MSE Reconstrucción")
ax.xaxis.set_major_formatter(FMT_DATE)
ax.legend(loc="upper left", fontsize=8, ncol=3,
          facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True)
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUTS_DIR, "G3_error_gru.png"),
             dpi=150, bbox_inches="tight", facecolor=BG)
print("[✓] G3_error_gru.png guardado")
plt.show(block=False)

# ── 2.9 Gráfico 4: Predicción vs Real — zoom por evento ──────────
y_pred_rs = y_pred_flat.reshape(-1, PRED_LEN, N_VARS)
y_all_rs  = y_all_flat.reshape(-1,  PRED_LEN, N_VARS)

# Media por ventana → desnormalizar TOIL_ENG (col 0)
def denorm_col(arr_3d, col_idx):
    means = arr_3d[:, :, col_idx].mean(axis=1)
    dummy = np.zeros((len(means), N_VARS))
    dummy[:, col_idx] = means
    return scaler.inverse_transform(dummy)[:, col_idx]

toil_pred = pd.Series(denorm_col(y_pred_rs, 0), index=ts_pred_dt)
toil_real = pd.Series(denorm_col(y_all_rs,  0), index=ts_pred_dt)

fig4, axes = plt.subplots(2, 2, figsize=(18, 10),
                           num="G4 — Predicción vs Real")
fig4.suptitle("GRU — Predicción vs Real de TOIL_ENG  |  ±6h por Evento",
              fontsize=12, fontweight="bold")
axes = axes.flatten()
for ax, (label, t_stop), ev_col in zip(axes, SHUTDOWNS.items(),
                                        EV_COLORS.values()):
    t0_z  = t_stop - pd.Timedelta(hours=6)
    t1_z  = t_stop + pd.Timedelta(hours=2)
    mz    = (toil_pred.index >= t0_z) & (toil_pred.index <= t1_z)
    ax.plot(toil_real[mz].index, toil_real[mz].values,
            color=ORANGE, lw=1.8, label="Real")
    ax.plot(toil_pred[mz].index, toil_pred[mz].values,
            color=ACCENT, lw=1.8, ls="--", label="GRU Predicción")
    ax.fill_between(toil_real[mz].index,
                    toil_real[mz].values, toil_pred[mz].values,
                    color=RED, alpha=0.20, label="Error")
    ax.axvline(t_stop, color=RED, lw=2.5, alpha=0.9)
    ax.text(t_stop, ax.get_ylim()[1] if ax.get_ylim()[1] else 200,
            " ⚡", color=RED, fontsize=12, va="top")
    ax.set_title(f"📅 {label}  |  {t_stop.strftime('%H:%M')}",
                 fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(FMT_HOUR)
    ax.set_ylabel("TOIL_ENG (°C)")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER,
              labelcolor=TEXT)
    ax.grid(True)
plt.tight_layout()
fig4.savefig(os.path.join(OUTPUTS_DIR, "G4_pred_vs_real.png"),
             dpi=150, bbox_inches="tight", facecolor=BG)
print("[✓] G4_pred_vs_real.png guardado")
plt.show(block=False)

# =============================================================================
# MÓDULO 3 — DETECCIÓN DE ANOMALÍAS Z-SCORE
# =============================================================================
print("\n" + "="*65)
print("  MÓDULO 3: DETECCIÓN DE ANOMALÍAS — Z-SCORE")
print("="*65)

# ── 3.1 Baseline stats y Z-score rodante ─────────────────────────
bstats   = df_base[VARIABLES].agg(["mean", "std"])
WINDOW   = 60
Z_YELLOW = 3.0
Z_RED    = 6.0

for col in VARIABLES:
    df[f"{col}_roll"] = df[col].rolling(WINDOW, min_periods=1).mean()
    mu = bstats.loc["mean", col]
    sg = bstats.loc["std",  col] + 1e-9
    df[f"{col}_z"] = (df[f"{col}_roll"] - mu) / sg

df["z_max"]  = df[[f"{c}_z" for c in VARIABLES]].abs().max(axis=1)
df["alerta"] = pd.cut(df["z_max"],
                       bins=[-np.inf, Z_YELLOW, Z_RED, np.inf],
                       labels=["Normal", "Amarilla", "Roja"])

n_amar = (df["alerta"] == "Amarilla").sum()
n_roja = (df["alerta"] == "Roja").sum()
print(f"[✓] Alertas Amarillas (|Z|>{Z_YELLOW:.0f}) : {n_amar:,}")
print(f"[✓] Alertas Rojas     (|Z|>{Z_RED:.0f})   : {n_roja:,}")

# ── 3.2 Primera alarma pre-parada ─────────────────────────────────
PRE_H = 30
print("\n--- Primera alarma antes de cada parada (ventana 30h) ---")
for label, t_stop in SHUTDOWNS.items():
    t0_win = t_stop - pd.Timedelta(hours=PRE_H)
    mask_w = (df["TIMESTAMP"] >= t0_win) & (df["TIMESTAMP"] < t_stop)
    sub    = df[mask_w].copy()
    sub["t_hrs"] = (sub["TIMESTAMP"] - t_stop).dt.total_seconds() / 3600
    print(f"\n  [{label}]  Parada: {t_stop.strftime('%H:%M')}")
    for col in ["TOIL_ENG", "POIL_ENG", "PSUCC", "PDESC", "RPM"]:
        alarma = sub[sub[f"{col}_z"].abs() > Z_YELLOW]["t_hrs"]
        if len(alarma):
            print(f"    {col:<12} → primera alarma "
                  f"{alarma.iloc[0]:+.1f}h  "
                  f"({abs(alarma.iloc[0]):.0f}h antes)")

# ── 3.3 Gráfico 5: Z-Score precursores serie completa ─────────────
fig5, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                           num="G5 — Z-Score Precursores")
fig5.suptitle("K7 — Z-Score Precursores Clave | Serie Completa",
              fontsize=12, fontweight="bold")
for ax, col, c in zip(axes, ["TOIL_ENG", "POIL_ENG"], [ORANGE, GREEN]):
    ax.plot(df["TIMESTAMP"], df[f"{col}_z"],
            color=c, lw=0.6, alpha=0.8, label=col)
    ax.axhline( Z_YELLOW, color=YELLOW, lw=0.9, ls="--", alpha=0.8)
    ax.axhline( Z_RED,    color=RED,    lw=0.9, ls="--", alpha=0.8,
                label=f"|Z|={Z_RED:.0f}")
    ax.axhline(-Z_YELLOW, color=YELLOW, lw=0.9, ls="--", alpha=0.8)
    ax.axhline(-Z_RED,    color=RED,    lw=0.9, ls="--", alpha=0.8)
    ax.fill_between(df["TIMESTAMP"], df[f"{col}_z"],
                    where=(df[f"{col}_z"].abs() > Z_RED),
                    color=RED, alpha=0.25, label="Zona crítica")
    for lbl, t_stop in SHUTDOWNS.items():
        ax.axvline(t_stop, color=EV_COLORS[lbl],
                   lw=1.3, ls="--", alpha=0.85, label=lbl)
    ax.set_ylabel(f"Z-Score {col}"); ax.grid(True)
    ax.legend(loc="upper left", fontsize=7, ncol=5,
              facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
axes[-1].xaxis.set_major_formatter(FMT_DATE)
plt.tight_layout()
fig5.savefig(os.path.join(OUTPUTS_DIR, "G5_zscore_serie.png"),
             dpi=150, bbox_inches="tight", facecolor=BG)
print("\n[✓] G5_zscore_serie.png guardado")
plt.show(block=False)

# ── 3.4 Gráfico 6: Línea de tiempo de alarmas ─────────────────────
ALARM_VARS = ["TOIL_ENG","POIL_ENG","PSUCC","PDESC",
              "TDESC_CIL1","TDESC_CIL2","TDESC_CIL3","TDESC_CIL4",
              "RPM","TOIL_COMP"]
ACOLORS    = [ORANGE, GREEN, YELLOW, "#3498db",
              "#e74c3c","#c0392b","#8e44ad","#16a085",
              RED, PURPLE]

fig6, axes = plt.subplots(2, 2, figsize=(20, 12),
                           num="G6 — Línea de Tiempo Alarmas")
fig6.suptitle("K7 — Línea de Tiempo de Alarmas Z-Score\n"
              "🟡 |Z|>3 Alerta   🔴 |Z|>6 Crítico   ⚡ Parada",
              fontsize=13, fontweight="bold", y=1.01)
axes = axes.flatten()

for ax, (label, t_stop) in zip(axes, SHUTDOWNS.items()):
    t0_win = t_stop - pd.Timedelta(hours=PRE_H)
    mask_w = (df["TIMESTAMP"] >= t0_win) & \
             (df["TIMESTAMP"] <= t_stop + pd.Timedelta(minutes=10))
    sub = df[mask_w].copy()
    sub["t_hrs"] = (sub["TIMESTAMP"] - t_stop).dt.total_seconds() / 3600
    n = len(ALARM_VARS)
    for i, (col, ac) in enumerate(zip(ALARM_VARS, ACOLORS)):
        y_pos = n - 1 - i
        ax.barh(y_pos, PRE_H, left=-PRE_H, height=0.6,
                color="#1c2128", alpha=0.8, zorder=1)
        m_y = sub[f"{col}_z"].abs() > Z_YELLOW
        for _, g in sub[m_y].groupby((~m_y).cumsum()):
            ax.barh(y_pos,
                    g["t_hrs"].iloc[-1] - g["t_hrs"].iloc[0],
                    left=g["t_hrs"].iloc[0], height=0.55,
                    color=YELLOW, alpha=0.55, zorder=2)
        m_r = sub[f"{col}_z"].abs() > Z_RED
        for _, g in sub[m_r].groupby((~m_r).cumsum()):
            ax.barh(y_pos,
                    g["t_hrs"].iloc[-1] - g["t_hrs"].iloc[0],
                    left=g["t_hrs"].iloc[0], height=0.55,
                    color=RED, alpha=0.80, zorder=3)
        if m_y.any():
            first = sub[m_y]["t_hrs"].iloc[0]
            ax.plot(first, y_pos, "|", color=YELLOW,
                    markersize=12, mew=2, zorder=4)
            ax.text(first - 0.3, y_pos, f"{abs(first):.0f}h",
                    color=YELLOW, fontsize=6.5,
                    ha="right", va="center")
    ax.axvline(0, color=RED, lw=3, alpha=0.95, zorder=5)
    ax.text(0.3, n - 0.4, "⚡ PARADA",
            color=RED, fontsize=9, fontweight="bold", va="top")
    ax.axvspan(-6, 0,       color=RED,    alpha=0.06)
    ax.axvspan(-PRE_H, -6,  color=ORANGE, alpha=0.04)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(ALARM_VARS)), fontsize=8)
    ax.set_xticks(range(-PRE_H, 1, 3))
    ax.set_xticklabels([f"{h}h" for h in range(-PRE_H, 1, 3)],
                        fontsize=7)
    ax.set_xlabel("Horas antes de la parada")
    ax.set_xlim(-PRE_H - 1, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_title(f"📅 {label}  |  {t_stop.strftime('%d/%b  %H:%M')}",
                 fontsize=10, fontweight="bold")
    ax.grid(True, axis="x")

leg = [mpatches.Patch(color=YELLOW, alpha=0.7, label=f"|Z|>{Z_YELLOW:.0f} Alerta"),
       mpatches.Patch(color=RED,    alpha=0.8, label=f"|Z|>{Z_RED:.0f} Crítico")]
fig6.legend(handles=leg, loc="lower center", ncol=2, fontsize=9,
            facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
            bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig6.savefig(os.path.join(OUTPUTS_DIR, "G6_alarmas_zscore.png"),
             dpi=150, bbox_inches="tight", facecolor=BG)
print("[✓] G6_alarmas_zscore.png guardado")
plt.show(block=False)

# ── 3.5 Gráfico 7: Boxplot normal vs pre-parada ───────────────────
BOX_VARS = ["TOIL_ENG","POIL_ENG","PSUCC","PDESC",
            "TDESC_CIL1","TDESC_CIL2","RPM","TOIL_COMP"]
fig7, axes = plt.subplots(2, 4, figsize=(20, 9),
                           num="G7 — Boxplot Normal vs Pre-Parada")
fig7.suptitle("K7 — Distribución: Operación Normal vs 30h Pre-Parada",
              fontsize=12, fontweight="bold")
axes = axes.flatten()
for ax, col in zip(axes, BOX_VARS):
    all_data   = [df_base[col].dropna().values]
    all_labels = ["Normal"]
    all_colors = [GREEN]
    for lbl, t_stop in SHUTDOWNS.items():
        t0_w   = t_stop - pd.Timedelta(hours=PRE_H)
        mask_w = (df["TIMESTAMP"] >= t0_w) & (df["TIMESTAMP"] <= t_stop)
        all_data.append(df[mask_w][col].dropna().values)
        all_labels.append(lbl)
        all_colors.append(EV_COLORS[lbl])
    bp = ax.boxplot(all_data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color=BORDER, lw=1),
                    capprops=dict(color=BORDER, lw=1),
                    flierprops=dict(marker=".", color=BORDER,
                                   markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(BORDER)
    ax.set_xticks(range(1, len(all_labels) + 1))
    ax.set_xticklabels(all_labels, fontsize=7, rotation=20)
    ax.set_title(col, fontsize=9, fontweight="bold", pad=4)
    ax.grid(True, axis="y")
plt.tight_layout()
fig7.savefig(os.path.join(OUTPUTS_DIR, "G7_boxplot.png"),
             dpi=150, bbox_inches="tight", facecolor=BG)
print("[✓] G7_boxplot.png guardado")
plt.show(block=False)

# ── 3.6 Gráfico 8: Cascada 30h × 4 eventos ───────────────────────
PHYS_VARS = ["TOIL_ENG","POIL_ENG","PSUCC","PDESC","TDESC_CIL1","RPM"]
PHYS_COLS = [ORANGE, GREEN, YELLOW, ACCENT, RED, "#58a6ff"]
PHYS_LBL  = ["T° Aceite Motor\n(°C)","P. Aceite Motor\n(psi)",
              "P. Succión\n(psi)","P. Descarga\n(psi)",
              "T° Desc CIL1\n(°C)","RPM"]

for label, t_stop in SHUTDOWNS.items():
    t0_win = t_stop - pd.Timedelta(hours=PRE_H)
    mask_w = (df["TIMESTAMP"] >= t0_win) & \
             (df["TIMESTAMP"] <= t_stop + pd.Timedelta(minutes=30))
    sub = df[mask_w].copy()
    fig, axes = plt.subplots(len(PHYS_VARS), 1, figsize=(18, 14),
                              sharex=True,
                              num=f"G8 Cascada {label}")
    fig.suptitle(
        f"K7 — Cascada de Variables Físicas  |  {label}\n"
        f"Parada: {t_stop.strftime('%H:%M del %d/%b/%Y')}  "
        f"| Ventana: {PRE_H}h",
        fontsize=12, fontweight="bold")
    for ax, col, c, ylbl in zip(axes, PHYS_VARS, PHYS_COLS, PHYS_LBL):
        ax.plot(sub["TIMESTAMP"], sub[col],
                color=c, lw=0.5, alpha=0.35)
        ax.plot(sub["TIMESTAMP"], sub[f"{col}_roll"],
                color=c, lw=1.8, alpha=0.95, label=col)
        mu_b = bstats.loc["mean", col]
        sg_b = bstats.loc["std",  col]
        ax.axhline(mu_b, color="white", lw=0.6,
                   ls="--", alpha=0.30)
        ax.axhspan(mu_b - 2*sg_b, mu_b + 2*sg_b,
                   color="white", alpha=0.03)
        ax.axvspan(t_stop - pd.Timedelta(hours=6), t_stop,
                   color=RED, alpha=0.08)
        ax.axvspan(t_stop - pd.Timedelta(hours=1), t_stop,
                   color=RED, alpha=0.14)
        ax.axvline(t_stop, color=RED, lw=2.5, alpha=0.9)
        for h_m in [24, 12, 6, 1]:
            t_m = t_stop - pd.Timedelta(hours=h_m)
            if t_m >= t0_win:
                ax.axvline(t_m, color="white",
                           lw=0.5, ls=":", alpha=0.25)
        ax.set_ylabel(ylbl, fontsize=8)
        ax.grid(True)
    axes[-1].xaxis.set_major_formatter(FMT_HOUR)
    axes[-1].set_xlabel("Hora del día")
    plt.tight_layout()
    fname = f"G8_cascada_{label.replace('/','')}.png"
    fig.savefig(os.path.join(OUTPUTS_DIR, fname),
                dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[✓] {fname} guardado")
    plt.show(block=False)

# =============================================================================
# MÓDULO 4 — REPORTE PDF
# =============================================================================
print("\n" + "="*65)
print("  MÓDULO 4: GENERANDO REPORTE PDF")
print("="*65)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rlcolors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImg, Table, TableStyle,
                                 PageBreak, HRFlowable)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

PDF_PATH = os.path.join(OUTPUTS_DIR, "K7_Reporte_Final.pdf")
doc = SimpleDocTemplate(PDF_PATH, pagesize=A4,
                         leftMargin=2*cm, rightMargin=2*cm,
                         topMargin=2*cm, bottomMargin=2*cm)

def ps(name, **kw):
    return ParagraphStyle(name,
                          fontName=kw.pop("fontName","Helvetica"),
                          fontSize=kw.pop("fontSize",9),
                          leading=kw.pop("leading",12),
                          **kw)

C_BLUE   = rlcolors.HexColor("#58a6ff")
C_PANEL  = rlcolors.HexColor("#161b22")
C_DARK   = rlcolors.HexColor("#0d1117")
C_BORDER = rlcolors.HexColor("#30363d")
C_TEXT   = rlcolors.HexColor("#c9d1d9")
C_GRAY   = rlcolors.HexColor("#8b949e")
C_BLUE2  = rlcolors.HexColor("#1f6feb")
C_ORANGE = rlcolors.HexColor("#e67e22")

S_T  = ps("T",  fontSize=20, textColor=C_BLUE,   alignment=TA_CENTER,
          spaceAfter=6, fontName="Helvetica-Bold")
S_S  = ps("S",  fontSize=13, textColor=rlcolors.white, alignment=TA_CENTER,
          spaceAfter=4, fontName="Helvetica-Bold")
S_M  = ps("M",  fontSize=10, textColor=C_GRAY,   alignment=TA_CENTER,
          spaceAfter=4)
S_H1 = ps("H1", fontSize=13, textColor=C_BLUE,   spaceBefore=12,
          spaceAfter=4, fontName="Helvetica-Bold")
S_H2 = ps("H2", fontSize=10, textColor=C_ORANGE, spaceBefore=8,
          spaceAfter=3, fontName="Helvetica-Bold")
S_B  = ps("B",  fontSize=9,  textColor=C_TEXT,   spaceAfter=4,
          leading=14, alignment=TA_JUSTIFY)
S_BU = ps("BU", fontSize=9,  textColor=C_TEXT,   spaceAfter=3,
          leftIndent=12, leading=13)
S_C  = ps("C",  fontSize=8,  textColor=rlcolors.HexColor("#2ecc71"),
          spaceAfter=3, fontName="Courier", leading=12)
S_CP = ps("CP", fontSize=8,  textColor=C_GRAY,   alignment=TA_CENTER,
          spaceAfter=6, fontName="Helvetica-Oblique")

BASE_TS = TableStyle([
    ("BACKGROUND",    (0,0), (-1,0),  C_BLUE2),
    ("TEXTCOLOR",     (0,0), (-1,0),  rlcolors.white),
    ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 8.5),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_PANEL, rlcolors.HexColor("#1c2128")]),
    ("TEXTCOLOR",     (0,1), (-1,-1), C_TEXT),
    ("GRID",          (0,0), (-1,-1), 0.3, C_BORDER),
    ("ALIGN",         (1,0), (-1,-1), "CENTER"),
    ("LEFTPADDING",   (0,0), (-1,-1), 7),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
])

def hr(): return HRFlowable(width="100%", thickness=0.5,
                              color=C_BORDER, spaceAfter=6)
def img(fname, w=15.5*cm, ratio=0.55):
    p = os.path.join(OUTPUTS_DIR, fname)
    return RLImg(p, width=w, height=w*ratio) if os.path.exists(p) \
           else Paragraph(f"[imagen: {fname}]", S_B)

story = []

# Portada
story += [
    Spacer(1, 2*cm),
    Paragraph("PROYECTO FINAL", S_T),
    Paragraph("Python para Analista de Datos — Industria Petrolera", S_S),
    Spacer(1, 0.5*cm), hr(),
    Paragraph("Sistema de Análisis y Detección de Anomalías", S_S),
    Paragraph("Equipo Compresor K7  |  Sep–Nov 2025", S_M),
    Spacer(1, 1.2*cm),
]
info = Table([
    ["Equipo",              "Compresor K7"],
    ["Período",             "01/Sep – 30/Nov 2025"],
    ["Registros",           f"{len(df):,}  (frecuencia 1 minuto)"],
    ["Variables SCADA",     str(len(VARIABLES))],
    ["Modelo",              "GRU 2 capas — TensorFlow / Keras"],
    ["Épocas entrenadas",   str(ep_real)],
    ["MAE modelo",          f"{mae_val:.4f}"],
    ["RMSE modelo",         f"{rmse_val:.4f}"],
    ["Alertas Amarillas",   f"{n_amar:,}"],
    ["Alertas Rojas",       f"{n_roja:,}"],
    
], colWidths=[5.5*cm, 10.5*cm])
info.setStyle(BASE_TS)
# Sección 0
story += [
    Paragraph(" Objetivo", S_H1), hr(),
    Paragraph(
        "El equipo a analizar es un motor a gas Waukesha VHP P-9390 acoplado "
        "a un compresor Dresser Rand. La data corresponde a 3 meses, en ese "
        "tiempo se identifico 4 paros no programados. las fechas correspondientes "
        "a esos paros se introducen en programa para que a partir de esos puntos "
        "se identifique patrones previos al paro con el obejtivo de anticiparce "
        "a esos eventos. ", S_B),
    Paragraph("Estadística descriptiva:", S_H2),
]
story += [info, Spacer(1, 1*cm), PageBreak()]

# Sección 1
story += [
    Paragraph("1. Ingesta, Limpieza y Estadística", S_H1), hr(),
    Paragraph(
        "Se cargaron 132,481 registros minuto a minuto del compresor K7 "
        "(Sep–Nov 2025). El CSV usa separador punto y coma y decimal con coma. "
        "La columna 15 vacía fue eliminada, los espacios en nombres de columnas "
        "reemplazados por guion bajo y los 155 valores nulos imputados por "
        "interpolación lineal.", S_B),
    Paragraph("Estadística descriptiva:", S_H2),
]
dr = [["Variable","Media","Std","Mín","Máx"]]
for col in ["RPM","PSUCC","PDESC","TOIL_ENG","POIL_ENG",
            "TDESC_CIL1","TDESC_CIL2","TDESC_CIL3","TDESC_CIL4"]:
    r = desc[col]
    dr.append([col, f"{r['mean']:.2f}", f"{r['std']:.2f}",
                f"{r['min']:.2f}", f"{r['max']:.2f}"])
dt = Table(dr, colWidths=[4*cm,3*cm,3*cm,2.5*cm,2.5*cm])
dt.setStyle(BASE_TS)
story += [dt, Spacer(1, 0.3*cm),
          img("G1_series_temporales.png"),
          Paragraph("Figura 1 — Series temporales Sep–Nov 2025.", S_CP),
          PageBreak()]

# Sección 2
story += [
    Paragraph("2. Modelo Predictivo — GRU (TensorFlow / Keras)", S_H1), hr(),
    Paragraph(
        "Red GRU de 2 capas entrenada exclusivamente con datos del período normal "
        "(Sep 1–9). Aprende el patrón de TOIL_ENG y POIL_ENG. "
        "El error de reconstrucción MSE sobre la serie completa actúa como "
        "detector: picos sobre μ+3σ indican anomalía. "
        "Se usaron EarlyStopping y ReduceLROnPlateau para evitar sobreajuste.", S_B),
    Paragraph("Arquitectura:", S_H2),
]
ar = [
    ["Parámetro","Valor"],
    ["Variables objetivo","TOIL_ENG, POIL_ENG"],
    ["Ventana de entrada",f"{SEQ_LEN} minutos"],
    ["Horizonte predicción",f"{PRED_LEN} minutos"],
    ["Arquitectura",
     "GRU(64)→Drop(0.2)→GRU(32)→Drop(0.2)→Dense(64,relu)→Dense(output)"],
    ["Épocas efectivas",str(ep_real)],
    ["Optimizador","Adam"],
    ["MAE",f"{mae_val:.4f}"],
    ["RMSE",f"{rmse_val:.4f}"],
    ["Umbral anomalía",f"{thr_gru:.6f}"],
    ["Anomalías GRU",f"{gru_anomaly.sum():,}"],
]
at = Table(ar, colWidths=[5.5*cm, 10.5*cm])
at.setStyle(BASE_TS)
story += [at, Spacer(1, 0.3*cm),
          img("G2_curvas_gru.png"),
          Paragraph("Figura 2 — Curvas de pérdida MSE y MAE por época.", S_CP),
          img("G3_error_gru.png"),
          Paragraph("Figura 3 — Error de reconstrucción GRU | Serie completa.", S_CP),
          img("G4_pred_vs_real.png"),
          Paragraph("Figura 4 — GRU: Predicción vs Real de TOIL_ENG ±6h por evento.", S_CP),
          PageBreak()]

# Sección 3
story += [
    Paragraph("3. Detección de Anomalías — Z-Score", S_H1), hr(),
    Paragraph(
        "Z-Score rodante (ventana 60 min) respecto al baseline (Sep 1–9). "
        "|Z|>3 → alerta amarilla. |Z|>6 → alerta roja. "
        "TOIL_ENG y POIL_ENG son detectables hasta 72h antes de la parada.", S_B),
]
alr = [["Evento","Parada","Precursor principal","Primera señal"]]
for lbl, h, p, a in [
    ("14/SEP","11:46","POIL_ENG","≥30h antes"),
    ("20/OCT","17:35","TOIL_ENG","≥30h antes"),
    ("31/OCT","02:51","TOIL_ENG + POIL_ENG","≥30h antes"),
    ("11/NOV","10:02","TOIL_ENG + POIL_ENG","≥30h antes"),
]:
    alr.append([lbl,h,p,a])
alt = Table(alr, colWidths=[2.5*cm,2.5*cm,7*cm,4*cm])
alt.setStyle(BASE_TS)
story += [alt, Spacer(1, 0.3*cm),
          img("G5_zscore_serie.png"),
          Paragraph("Figura 5 — Z-Score TOIL_ENG y POIL_ENG | Serie completa.", S_CP),
          img("G6_alarmas_zscore.png"),
          Paragraph("Figura 6 — Línea de tiempo de alarmas Z-Score por evento.", S_CP),
          img("G7_boxplot.png"),
          Paragraph("Figura 7 — Distribución normal vs 30h pre-parada.", S_CP),
          PageBreak()]

# Sección 4
story += [
    Paragraph("4. Cascada de Variables por Evento", S_H1), hr(),
    Paragraph("Evolución de variables críticas 30h antes de cada parada.", S_B),
]
for label in LABELS:
    fname = f"G8_cascada_{label.replace('/','')}.png"
    story += [img(fname),
              Paragraph(f"Figura 8{label} — Cascada | {label}", S_CP)]
story.append(PageBreak())

# Sección 5
story += [
    Paragraph("5. Conclusiones y Recomendaciones", S_H1), hr(),
    Paragraph("<b>Variables precursoras:</b>", S_H2),
    Paragraph("• TOIL_ENG y POIL_ENG detectables hasta 72h antes en 3/4 eventos.", S_BU),
    Paragraph("• RPM y PSUCC colapsan al final — efecto, no causa.", S_BU),
    Paragraph("• 31/OCT fue el evento más severo (ΔRPM=−899).", S_BU),
    Spacer(1, 0.3*cm),
    Paragraph("<b>Diagnóstico por evento:</b>", S_H2),
    Paragraph("• 14/SEP: POIL_ENG cae 72h antes → lubricación insuficiente → trip.", S_BU),
    Paragraph("• 20/OCT: TOIL_ENG Z=15.8 por 72h → sobrecalentamiento aceite.", S_BU),
    Paragraph("• 31/OCT: TOIL+POIL elevados 72h + colapso PSUCC → doble falla.", S_BU),
    Paragraph("• 11/NOV: Patrón idéntico a 20/OCT → falla recurrente enfriador.", S_BU),
    Paragraph("==================================================================", S_BU),
    Paragraph("<b>Información Real Reportado por personal de mtto:</b>", S_H2),
    Paragraph("• 14/SEP: PROBLEMA DE BAJA PRESION DE ACEITE. SE REVISO BBA, LINEAS, FILTROS ", S_BU),
    Paragraph("          DURANTE UNA SEMANA, SE LOGRO INCREMENTAR PRESION HASTA 43 PSI SIENDO LO NORMAL 50-55 PSI ", S_BU),
    Paragraph("          EN ESTE CASO LA SEÑAL ES UN DECREMENTO CONTINUO DE LA PRESION DE ACEITE MOTOR ", S_BU),    
    Paragraph("• 20/OCT: FALLA SENSOR DE TEMPERATURA ACEITE MOTOR.", S_BU),
    Paragraph("          EN ESTE CASO ES FALLA DE SENSOR QUE LO IDENTIFICA EL ANÁLISIS PERO NO HAY OTRAS VARIABLES AFECTADAS ", S_BU),   
    Paragraph("• 31/OCT: CORREA BOMBA DE REFRIGERACION EN MAS ESTADO.", S_BU),
    Paragraph("          EN ESTE CASO LA FALLA EN LA CORREA AFECTO A OTRAS VARIABLES DEL MOTOR ", S_BU),
    Paragraph("• 11/NOV: ROTURA SOPORTE BBA PRINCIPAL AGUA DE REFRIGERACIÓN.", S_BU),
    Paragraph("          EN ESTE CASO DE IGUAL FORMA LA FALLA AFECTO A OTRAS VARIABLES DEL MOTOR ", S_BU),
    Spacer(1, 0.3*cm),
    Paragraph("<b>Regla de alarma temprana:</b>", S_H2),
    Paragraph(
        "ALERTA AMARILLA cuando |Z(TOIL_ENG)| > 3 por ≥60 min. "
        "ALERTA ROJA cuando |Z(TOIL_ENG)| > 6 Y |Z(POIL_ENG)| > 3 por ≥30 min. "
        "Habría detectado 3/4 eventos con >12h de anticipación.", S_B),
    Spacer(1, 0.3*cm),
    Paragraph("<b>Métricas GRU (TensorFlow):</b>", S_H2),
    Paragraph(
        f"MAE = {mae_val:.4f}  |  RMSE = {rmse_val:.4f}  "
        f"|  Épocas = {ep_real}  |  Umbral = {thr_gru:.6f}", S_C),
    Spacer(1, 0.5*cm), hr(),
    Paragraph(
        "Proyecto Final — Python para Analista de Datos en la Industria Petrolera",
        ps("F", fontSize=8, textColor=C_GRAY, alignment=TA_CENTER)),
]

doc.build(story)
print(f"[✓] PDF generado: {PDF_PATH}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*65)
print("  PROCESO COMPLETADO ✓")
print("="*65)
print(f"  Registros procesados     : {len(df):,}")
print(f"  Épocas entrenamiento GRU : {ep_real}")
print(f"  MAE  modelo GRU          : {mae_val:.4f}")
print(f"  RMSE modelo GRU          : {rmse_val:.4f}")
print(f"  Umbral anomalía GRU      : {thr_gru:.6f}")
print(f"  Anomalías GRU            : {gru_anomaly.sum():,}")
print(f"  Alertas Amarillas (Z>3)  : {n_amar:,}")
print(f"  Alertas Rojas     (Z>6)  : {n_roja:,}")
print(f"\n  Archivos en : {OUTPUTS_DIR}")
for f in sorted(os.listdir(OUTPUTS_DIR)):
    kb = os.path.getsize(os.path.join(OUTPUTS_DIR, f)) / 1024
    print(f"    {f:<38} {kb:>7.1f} KB")
print(f"\n  Modelo guardado: {MODEL_PATH}")
print("="*65)
