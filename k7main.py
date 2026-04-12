"""
====================================================================
PROYECTO FINAL — Equipo K7  |  Sep–Nov 2025
Análisis de Señales PRECURSORAS antes de Paradas Bruscas
Objetivo: identificar qué variable se deteriora ANTES de la falla
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ── Paleta visual ────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#c9d1d9"
ACCENT  = "#58a6ff"
RED     = "#ff4444"
ORANGE  = "#e67e22"
GREEN   = "#2ecc71"
YELLOW  = "#f1c40f"

COLORES_EVENTO = {
    "14/SEP": "#e74c3c",
    "20/OCT": "#e67e22",
    "31/OCT": "#9b59b6",
    "11/NOV": "#1abc9c",
}

# ─────────────────────────────────────────────────────────────────
# 1. INGESTA Y LIMPIEZA
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  PASO 1: INGESTA Y LIMPIEZA")
print("=" * 65)

df = pd.read_csv(
    "K7_SepOctNov.csv",
    sep=";", decimal=",", engine="python", on_bad_lines="skip"
)
df.columns = [c.replace(" ", "_") for c in df.columns]
col_vacia = df.columns[14]
df.drop(columns=[col_vacia], inplace=True)
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], dayfirst=True, errors="coerce")
df.sort_values("TIMESTAMP", inplace=True)
df.reset_index(drop=True, inplace=True)

variables = [c for c in df.columns if c != "TIMESTAMP"]
df[variables] = df[variables].interpolate(method="linear").ffill().bfill()

print(f"[✓] {len(df):,} registros  |  columnas: {variables}")

# ─────────────────────────────────────────────────────────────────
# 2. MOMENTOS EXACTOS DE PARADA (detección por caída de RPM)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PASO 2: DETECCIÓN DE MOMENTOS DE PARADA")
print("=" * 65)

FECHAS_PROBLEMA = ["2025-09-14", "2025-10-20", "2025-10-31", "2025-11-11"]
LABELS          = ["14/SEP", "20/OCT", "31/OCT", "11/NOV"]

SHUTDOWNS = {}
for fp, label in zip(FECHAS_PROBLEMA, LABELS):
    t0   = pd.Timestamp(fp)
    mask = (df["TIMESTAMP"] >= t0) & (df["TIMESTAMP"] < t0 + pd.Timedelta(hours=24))
    sub  = df[mask]
    # El momento de parada = mayor caída de RPM en el día
    rpm_diff  = sub["RPM"].diff()
    min_idx   = rpm_diff.idxmin()
    t_stop    = sub.loc[min_idx, "TIMESTAMP"]
    rpm_drop  = rpm_diff[min_idx]
    SHUTDOWNS[label] = t_stop
    print(f"  [{label}]  Parada detectada: {t_stop.strftime('%Y-%m-%d %H:%M')}  "
          f"|  ΔRPM = {rpm_drop:.0f}")

# ─────────────────────────────────────────────────────────────────
# 3. BASELINE Y Z-SCORE RODANTE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PASO 3: BASELINE NORMAL Y Z-SCORE RODANTE")
print("=" * 65)

# Baseline: primeros 10 días de Sep (período sin eventos)
baseline = df[df["TIMESTAMP"] < "2025-09-10"]
bstats   = baseline[variables].agg(["mean", "std"])

# Z-score con media rodante de 60 min (suaviza ruido de 1 min)
WINDOW = 60
for col in variables:
    df[f"{col}_roll"] = df[col].rolling(WINDOW, min_periods=1).mean()
    mu = bstats.loc["mean", col]
    sg = bstats.loc["std",  col] + 1e-9
    df[f"{col}_z"]   = (df[f"{col}_roll"] - mu) / sg

print(f"[✓] Baseline: {baseline['TIMESTAMP'].min().date()} → {baseline['TIMESTAMP'].max().date()}")
print(f"[✓] Z-score rodante (ventana {WINDOW} min) calculado para {len(variables)} variables")

# ─────────────────────────────────────────────────────────────────
# 4. ISOLATION FOREST — entrenado SOLO con datos basales
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PASO 4: MODELO ISOLATION FOREST (baseline training)")
print("=" * 65)

feat_cols  = [f"{c}_roll" for c in variables]
X_baseline = baseline[variables].values
scaler     = StandardScaler().fit(X_baseline)
X_all      = scaler.transform(df[variables].values)
X_base_sc  = scaler.transform(X_baseline)

iforest = IsolationForest(
    n_estimators=300,
    contamination=0.01,   # 1% anomalías en baseline
    max_samples=min(10000, len(X_base_sc)),
    random_state=42,
    n_jobs=-1,
)
iforest.fit(X_base_sc)

df["if_score"] = iforest.decision_function(X_all)   # negativo = más anómalo
df["if_label"] = iforest.predict(X_all)              # -1 = anomalía

print(f"[✓] Entrenado con {len(X_base_sc):,} muestras basales")

# ─────────────────────────────────────────────────────────────────
# 5. ANÁLISIS PRECURSOR por ventana pre-parada
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PASO 5: SEÑALES PRECURSORAS (ventanas antes de parada)")
print("=" * 65)

VENTANAS_H = [24, 12, 6, 3, 1]
precursor_report = {}

for label, t_stop in SHUTDOWNS.items():
    print(f"\n  ── {label} | Parada: {t_stop.strftime('%H:%M')} ──")
    evento = {}
    for h in VENTANAS_H:
        t_check = t_stop - pd.Timedelta(hours=h)
        row     = df[df["TIMESTAMP"] <= t_check].iloc[-1]
        z_vals  = {col: abs(row[f"{col}_z"]) for col in variables}
        top4    = sorted(z_vals.items(), key=lambda x: x[1], reverse=True)[:4]
        evento[h] = top4
        print(f"    -{h:2d}h:  " +
              "  ".join([f"{k}(Z={v:+.1f})" for k, v in top4]))
    precursor_report[label] = evento

# Variable dominante precursora por evento
print("\n  ── VARIABLE PRECURSORA DOMINANTE (−24h antes) ──")
for label, evento in precursor_report.items():
    top_var = evento[24][0][0]
    top_z   = evento[24][0][1]
    print(f"    {label}: {top_var}  (|Z|={top_z:.1f} a −24h de la parada)")

# ─────────────────────────────────────────────────────────────────
# 6. GRÁFICOS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PASO 6: GENERANDO GRÁFICOS PRECURSORES")
print("=" * 65)

DATE_FMT  = DateFormatter("%d-%b")
HOUR_FMT  = DateFormatter("%H:%M")
PRE_WINDOW = 30  # horas antes de la parada a mostrar

# ════════════════════════════════════════════════════════════════
# GRÁFICO 1 — Panel 4 eventos: Z-score de variables clave pre-parada
# ════════════════════════════════════════════════════════════════
KEY_VARS = ["TOIL_ENG", "POIL_ENG", "PSUCC", "TDESC_CIL1", "RPM"]
KEY_COLS = [ACCENT, ORANGE, GREEN, YELLOW, "#e74c3c"]

fig, axes = plt.subplots(2, 2, figsize=(20, 12), facecolor=BG)
fig.suptitle(
    "K7 — Señales Precursoras de Falla\n"
    "Z-Score de variables críticas en las horas previas a cada parada",
    color=TEXT, fontsize=14, fontweight="bold", y=1.01
)
axes = axes.flatten()

for ax, (label, t_stop), color_ev in zip(
        axes, SHUTDOWNS.items(), COLORES_EVENTO.values()):
    ax.set_facecolor(PANEL)

    t0_win = t_stop - pd.Timedelta(hours=PRE_WINDOW)
    mask   = (df["TIMESTAMP"] >= t0_win) & (df["TIMESTAMP"] <= t_stop + pd.Timedelta(hours=2))
    sub    = df[mask].copy()

    # Línea de umbral ±3σ
    ax.axhline( 3, color=BORDER, lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-3, color=BORDER, lw=0.8, ls="--", alpha=0.7)
    ax.axhline( 0, color=BORDER, lw=0.6, alpha=0.5)

    # Zona roja: 1h antes de parada
    ax.axvspan(t_stop - pd.Timedelta(hours=1), t_stop,
               color=RED, alpha=0.12, label="−1h crítica")

    # Zona naranja: 6h antes
    ax.axvspan(t_stop - pd.Timedelta(hours=6),
               t_stop - pd.Timedelta(hours=1),
               color=ORANGE, alpha=0.07, label="−6h a −1h")

    # Línea de parada
    ax.axvline(t_stop, color=RED, lw=2, ls="-", alpha=0.9,
               label=f"Parada {t_stop.strftime('%H:%M')}")

    # Z-score de cada variable clave
    for col, c in zip(KEY_VARS, KEY_COLS):
        if f"{col}_z" in sub.columns:
            ax.plot(sub["TIMESTAMP"], sub[f"{col}_z"],
                    color=c, lw=1.2, alpha=0.85, label=col)

    ax.set_title(f"📅 {label}  |  Parada: {t_stop.strftime('%H:%M')}",
                 color=TEXT, fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel("|Z-Score| vs Baseline", color="#8b949e", fontsize=9)
    ax.xaxis.set_major_formatter(HOUR_FMT if PRE_WINDOW <= 30 else DATE_FMT)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines[:].set_color(BORDER)
    ax.grid(True, color="#21262d", lw=0.5)
    ax.legend(loc="upper left", fontsize=7, ncol=2,
              facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_ylim(-5, 20)

plt.tight_layout()
plt.savefig("precursor1_zscore_previo.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("[✓] precursor1_zscore_previo.png")


# ════════════════════════════════════════════════════════════════
# GRÁFICO 2 — Serie completa: IF Score + líneas de parada
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(20, 13), facecolor=BG,
                         gridspec_kw={"height_ratios": [2, 1.5, 1.5]})
fig.suptitle("K7 — Isolation Forest Score + Variables Físicas Clave\n"
             "Vista completa Sep–Nov 2025",
             color=TEXT, fontsize=13, fontweight="bold")

# — Panel 1: IF Score
ax = axes[0]
ax.set_facecolor(PANEL)
ax.fill_between(df["TIMESTAMP"], df["if_score"], 0,
                where=(df["if_score"] < 0),
                color=RED, alpha=0.35, label="Zona anómala (IF Score < 0)")
ax.fill_between(df["TIMESTAMP"], df["if_score"], 0,
                where=(df["if_score"] >= 0),
                color=ACCENT, alpha=0.2, label="Zona normal")
ax.plot(df["TIMESTAMP"], df["if_score"], color=ACCENT, lw=0.3, alpha=0.5)
ax.axhline(0, color=RED, lw=0.8, ls="--")
for label, t_stop in SHUTDOWNS.items():
    col = COLORES_EVENTO[label]
    ax.axvline(t_stop, color=col, lw=1.5, ls="--", alpha=0.9, label=f"Parada {label}")
ax.set_ylabel("IF Score", color="#8b949e", fontsize=9)
ax.set_title("Anomaly Score (Isolation Forest) — negativo = anómalo",
             color=TEXT, fontsize=10, pad=4)
ax.legend(loc="lower left", fontsize=8, ncol=4,
          facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.tick_params(colors="#8b949e", labelsize=8)
ax.spines[:].set_color(BORDER)
ax.xaxis.set_major_formatter(DATE_FMT)
ax.grid(True, color="#21262d", lw=0.5)

# — Panel 2: TOIL_ENG y POIL_ENG (principales precursores térmicos/presión aceite)
ax = axes[1]
ax.set_facecolor(PANEL)
ax2 = ax.twinx(); ax2.set_facecolor(PANEL)
ax.plot(df["TIMESTAMP"], df["TOIL_ENG"], color=ORANGE, lw=0.5,
        alpha=0.7, label="TOIL_ENG (°C)")
ax2.plot(df["TIMESTAMP"], df["POIL_ENG"], color=GREEN, lw=0.5,
         alpha=0.7, label="POIL_ENG (psi)", ls="--")
for label, t_stop in SHUTDOWNS.items():
    ax.axvline(t_stop, color=COLORES_EVENTO[label], lw=1.3, ls="--", alpha=0.8)
ax.set_ylabel("TOIL_ENG (°C)", color=ORANGE, fontsize=9)
ax2.set_ylabel("POIL_ENG (psi)", color=GREEN, fontsize=9)
ax.set_title("Temperatura y Presión de Aceite — principales precursores",
             color=TEXT, fontsize=10, pad=4)
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=8, facecolor=PANEL,
          edgecolor=BORDER, labelcolor=TEXT, loc="upper left")
ax.tick_params(colors="#8b949e", labelsize=8)
ax2.tick_params(colors="#8b949e", labelsize=8)
ax.spines[:].set_color(BORDER); ax2.spines[:].set_color(BORDER)
ax.xaxis.set_major_formatter(DATE_FMT)
ax.grid(True, color="#21262d", lw=0.5)

# — Panel 3: PSUCC y RPM
ax = axes[2]
ax.set_facecolor(PANEL)
ax3 = ax.twinx(); ax3.set_facecolor(PANEL)
ax.plot(df["TIMESTAMP"], df["PSUCC"], color=YELLOW, lw=0.5,
        alpha=0.7, label="PSUCC (psi)")
ax3.plot(df["TIMESTAMP"], df["RPM"], color=ACCENT, lw=0.5,
         alpha=0.7, label="RPM")
for label, t_stop in SHUTDOWNS.items():
    ax.axvline(t_stop, color=COLORES_EVENTO[label], lw=1.3, ls="--", alpha=0.8)
ax.set_ylabel("PSUCC (psi)", color=YELLOW, fontsize=9)
ax3.set_ylabel("RPM", color=ACCENT, fontsize=9)
ax.set_title("Presión de Succión y RPM — colapso en el momento de parada",
             color=TEXT, fontsize=10, pad=4)
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax3.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=8, facecolor=PANEL,
          edgecolor=BORDER, labelcolor=TEXT, loc="upper left")
ax.tick_params(colors="#8b949e", labelsize=8)
ax3.tick_params(colors="#8b949e", labelsize=8)
ax.spines[:].set_color(BORDER); ax3.spines[:].set_color(BORDER)
ax.xaxis.set_major_formatter(DATE_FMT)
ax.grid(True, color="#21262d", lw=0.5)

plt.tight_layout()
plt.savefig("precursor2_serie_global.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("[✓] precursor2_serie_global.png")


# ════════════════════════════════════════════════════════════════
# GRÁFICO 3 — Cascada de variables físicas: 30h antes de parada
# ════════════════════════════════════════════════════════════════
PHYS_VARS  = ["TOIL_ENG", "POIL_ENG", "PSUCC", "PDESC", "TDESC_CIL1", "RPM"]
PHYS_COLS  = [ORANGE, GREEN, YELLOW, "#3498db", "#e74c3c", ACCENT]
PHYS_YLBL  = ["T Aceite Motor\n(°C)", "P Aceite Motor\n(psi)", "P Succión\n(psi)",
               "P Descarga\n(psi)", "T Desc CIL1\n(°C)", "RPM"]

for label, t_stop in SHUTDOWNS.items():
    color_ev = COLORES_EVENTO[label]
    t0_win   = t_stop - pd.Timedelta(hours=PRE_WINDOW)
    mask     = (df["TIMESTAMP"] >= t0_win) & \
               (df["TIMESTAMP"] <= t_stop + pd.Timedelta(minutes=30))
    sub      = df[mask].copy()

    fig, axes = plt.subplots(len(PHYS_VARS), 1,
                             figsize=(18, 14), facecolor=BG,
                             sharex=True)
    fig.suptitle(
        f"K7 — Cascada de Variables Físicas | Evento: {label}\n"
        f"Ventana: {PRE_WINDOW}h antes → parada a las {t_stop.strftime('%H:%M')}",
        color=TEXT, fontsize=13, fontweight="bold"
    )

    for ax, col, c, ylbl in zip(axes, PHYS_VARS, PHYS_COLS, PHYS_YLBL):
        ax.set_facecolor(PANEL)

        # Sombreados temporales
        ax.axvspan(t_stop - pd.Timedelta(hours=6), t_stop,
                   color=RED, alpha=0.08)
        ax.axvspan(t_stop - pd.Timedelta(hours=1), t_stop,
                   color=RED, alpha=0.15)
        ax.axvline(t_stop, color=RED, lw=2, ls="-", alpha=0.9)

        # Valor real + media rodante
        ax.plot(sub["TIMESTAMP"], sub[col],
                color=c, lw=0.5, alpha=0.5)
        ax.plot(sub["TIMESTAMP"], sub[f"{col}_roll"],
                color=c, lw=1.5, alpha=0.95, label=col)

        # Marcas en -24h, -12h, -6h, -1h
        for h_mark, h_alpha in [(24, 0.4), (12, 0.5), (6, 0.7), (1, 0.9)]:
            t_mark = t_stop - pd.Timedelta(hours=h_mark)
            if t_mark >= t0_win:
                ax.axvline(t_mark, color="#ffffff", lw=0.6,
                           ls=":", alpha=h_alpha)
                if ax == axes[0]:
                    ax.text(t_mark, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                            f"−{h_mark}h", color="#ffffff", fontsize=7,
                            ha="center", va="bottom", alpha=h_alpha)

        ax.set_ylabel(ylbl, color=c, fontsize=8)
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_color(BORDER)
        ax.grid(True, color="#21262d", lw=0.5)

    axes[-1].xaxis.set_major_formatter(HOUR_FMT)
    axes[-1].set_xlabel("Hora del día", color="#8b949e", fontsize=9)

    # Leyenda de zonas
    p1 = mpatches.Patch(color=RED, alpha=0.25, label="Zona crítica (−1h → parada)")
    p2 = mpatches.Patch(color=RED, alpha=0.1,  label="Zona alerta (−6h → −1h)")
    p3 = mpatches.Patch(color=RED, alpha=0.9,  label=f"Parada {t_stop.strftime('%H:%M')}")
    fig.legend(handles=[p1, p2, p3], loc="lower center",
               ncol=3, fontsize=9, facecolor=PANEL,
               edgecolor=BORDER, labelcolor=TEXT,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fname = f"precursor3_cascada_{label.replace('/', '')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[✓] {fname}")


# ════════════════════════════════════════════════════════════════
# GRÁFICO 4 — Heatmap: Z-Score de todas las variables × hora previa
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 11), facecolor=BG)
fig.suptitle("K7 — Heatmap Z-Score por Variable en las Horas Previas a Cada Parada",
             color=TEXT, fontsize=13, fontweight="bold")
axes = axes.flatten()

HORAS = list(range(-PRE_WINDOW, 1))  # -30h a 0

for ax, (label, t_stop) in zip(axes, SHUTDOWNS.items()):
    ax.set_facecolor(PANEL)
    matrix = np.zeros((len(variables), len(HORAS)))
    for j, h in enumerate(HORAS):
        t_check = t_stop + pd.Timedelta(hours=h)
        row     = df[df["TIMESTAMP"] <= t_check].iloc[-1]
        for i, col in enumerate(variables):
            matrix[i, j] = row[f"{col}_z"]

    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-5, vmax=12,
                   aspect="auto", interpolation="bilinear")
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables, color=TEXT, fontsize=8)
    ax.set_xticks(np.arange(0, len(HORAS), 6))
    ax.set_xticklabels([f"{HORAS[i]}h" for i in np.arange(0, len(HORAS), 6)],
                        color="#8b949e", fontsize=8)
    ax.set_title(f"📅 {label}  |  Parada: {t_stop.strftime('%H:%M')}",
                 color=TEXT, fontsize=10, fontweight="bold", pad=6)

    # Línea de parada (hora 0)
    ax.axvline(len(HORAS) - 1, color=RED, lw=2, alpha=0.9)
    ax.text(len(HORAS) - 1, -0.7, "PARADA", color=RED,
            fontsize=7, ha="center", fontweight="bold")
    ax.spines[:].set_color(BORDER)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label("Z-Score vs Baseline Normal", color=TEXT, fontsize=9)
cb.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

plt.subplots_adjust(right=0.90, hspace=0.35, wspace=0.15)
plt.savefig("precursor4_heatmap_zscore.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("[✓] precursor4_heatmap_zscore.png")


# ════════════════════════════════════════════════════════════════
# GRÁFICO 5 — Radar/barras: ranking de variables por Z-score
#              a distintas ventanas temporales
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=BG)
fig.suptitle("K7 — Ranking de Variables Precursoras por Ventana Temporal\n"
             "(cuanto mayor el Z-Score, antes avisa del problema)",
             color=TEXT, fontsize=13, fontweight="bold")
axes = axes.flatten()

VENTANAS_PLOT = [24, 12, 6, 1]
V_COLORS      = [ACCENT, GREEN, ORANGE, RED]

for ax, (label, t_stop) in zip(axes, SHUTDOWNS.items()):
    ax.set_facecolor(PANEL)

    x      = np.arange(len(variables))
    width  = 0.2
    bar_groups = []
    for k, (h, vc) in enumerate(zip(VENTANAS_PLOT, V_COLORS)):
        t_check = t_stop - pd.Timedelta(hours=h)
        row     = df[df["TIMESTAMP"] <= t_check].iloc[-1]
        z_vals  = [abs(row[f"{col}_z"]) for col in variables]
        bars = ax.bar(x + k * width, z_vals, width,
                      color=vc, alpha=0.75, label=f"−{h}h")
        bar_groups.append(bars)

    ax.axhline(3, color="white", lw=0.8, ls="--", alpha=0.5, label="Umbral Z=3")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(variables, rotation=40, ha="right",
                       color=TEXT, fontsize=8)
    ax.set_ylabel("|Z-Score|", color="#8b949e", fontsize=9)
    ax.set_title(f"📅 {label}  |  Parada: {t_stop.strftime('%H:%M')}",
                 color=TEXT, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines[:].set_color(BORDER)
    ax.grid(True, color="#21262d", lw=0.5, axis="y")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER,
              labelcolor=TEXT, loc="upper right")
    ax.set_ylim(0, max(20, ax.get_ylim()[1]))

plt.tight_layout()
plt.savefig("precursor5_ranking_variables.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("[✓] precursor5_ranking_variables.png")


# ─────────────────────────────────────────────────────────────────
# 7. REPORTE FINAL
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  REPORTE FINAL — DIAGNÓSTICO DE CAUSAS")
print("=" * 65)

diagnostico = {
    "14/SEP": {
        "t_stop": "11:46",
        "causa_probable": "Caída de POIL_ENG (presión aceite motor) persistente",
        "señal_mas_temprana": "POIL_ENG con Z=9.6 desde 24h antes",
        "mecanismo": "Baja presión de aceite → lubricación insuficiente → protección activa",
    },
    "20/OCT": {
        "t_stop": "17:35",
        "causa_probable": "TOIL_ENG disparado (temperatura aceite motor) + POIL_ENG elevado",
        "señal_mas_temprana": "TOIL_ENG con Z=15.8 sostenido desde 24h antes",
        "mecanismo": "Sobrecalentamiento de aceite → posible obstrucción filtro/enfriador",
    },
    "31/OCT": {
        "t_stop": "02:51",
        "causa_probable": "Colapso de PSUCC (presión succión) + TOIL_ENG + POIL_ENG elevados",
        "señal_mas_temprana": "TOIL_ENG(Z=11.3) + POIL_ENG(Z=6.6) desde 24h; PSUCC se desploma −1h",
        "mecanismo": "Problema de proceso en succión + estrés térmico acumulado → parada de emergencia",
    },
    "11/NOV": {
        "t_stop": "10:02",
        "causa_probable": "TOIL_ENG elevado sostenido + aumento gradual POIL_ENG",
        "señal_mas_temprana": "TOIL_ENG con Z=6.8–8.0 en las 24h previas",
        "mecanismo": "Patrón igual a 20/OCT: problema recurrente en sistema de aceite del motor",
    },
}

for label, info in diagnostico.items():
    print(f"\n  📅 {label}  |  Parada: {info['t_stop']}")
    print(f"     Causa probable       : {info['causa_probable']}")
    print(f"     Señal más temprana   : {info['señal_mas_temprana']}")
    print(f"     Mecanismo sugerido   : {info['mecanismo']}")

print(f"""
  ── PATRÓN COMÚN ──
  ▸ TOIL_ENG (Temperatura Aceite Motor) es el precursor más
    consistente y temprano: Z > 6 en 3 de 4 eventos, hasta 24h antes.
  ▸ POIL_ENG (Presión Aceite Motor) también anómalo en todos los eventos.
  ▸ RPM y PSUCC colapsan ÚLTIMO (son consecuencia, no causa).
  ▸ Recomendación: activar alarma temprana cuando
    TOIL_ENG_z > 4  Y  POIL_ENG_z > 2  por más de 30 min consecutivos.

  Gráficos generados:
    precursor1_zscore_previo.png       — Z-score por evento
    precursor2_serie_global.png        — Vista global + marcadores
    precursor3_cascada_[evento].png    — Cascada física (×4 eventos)
    precursor4_heatmap_zscore.png      — Heatmap temporal
    precursor5_ranking_variables.png   — Ranking por ventana temporal
""")
