# ============================================================
# Heat waves and monthly deaths – France, 95 départements,
# June–September, 2000–2025
#
# One script, three outputs (opened in windows, NOT saved by default):
#   Figure 1 : months with vs without at least one heat-wave day
#   Figure 2 : months by number of heat-wave days (0, 1–3, 4–6, 7–9, 10+)
#   Table    : regression table comparing the groups (all years / without 2003)
#
# Heat-wave day : canicule_official == 1 (official Météo-France flag)
#
# Dependent variable – deviation from a "standard summer", adjusted for
# population growth and ageing:
#   monthly_avg[d, y] = total deaths of département d in year y / 12
#   ratio[d, y, m]    = deaths[d, y, m] / monthly_avg[d, y]
#   standard[d, m]    = mean of ratio[d, ·, m] over the sample years
#   deviation (%)     = (ratio / standard - 1) * 100
#
# Regression columns
#   (1)/(4) At least one heat-wave day (dummy)
#   (2)/(5) Number-of-days bins (reference = 0 days)
#   (3)/(6) Bins + département FE + year×month FE
#   (1)–(3) all years, (4)–(6) without 2003. SE clustered by département.
#
# Put this file in the folder with Temperature_2000_2025.csv and
# DECES_2000_2025.csv (or set DATA_DIR) and run it.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Settings ------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
SAVE = False                                   # True = also save the PNGs in DATA_DIR/figures
GRAPH_SAMPLES = ["all", "no2003"]              # which samples to draw figures for: "all", "no2003"

SUMMER_MONTHS = [6, 7, 8, 9]
MONTH_NAMES = {6: "June", 7: "July", 8: "August", 9: "September"}
BINS = [-0.5, 0.5, 3.5, 6.5, 9.5, np.inf]
BIN_LABELS = ["0", "1–3", "4–6", "7–9", "10+"]
SAMPLES = {"all": ([], "2000–2025"), "no2003": ([2003], "2000–2025 without 2003")}

C_NO, C_HW = "#2a78d6", "#eb6834"              # blue = no heat wave, orange = heat wave
BIN_COLORS = ["#2a78d6", "#f6b89a", "#f08a5d", "#eb6834", "#b8461b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

# ---- 1. Read the data ----------------------------------------
temp_all = pd.read_csv(os.path.join(DATA_DIR, "Temperature_2000_2025.csv"), sep=";", decimal=",",
                       encoding="utf-8-sig", dtype={"dep": str})
deaths_all = pd.read_csv(os.path.join(DATA_DIR, "DECES_2000_2025.csv"), sep=";",
                         encoding="utf-8-sig", dtype={"dep": str})
for d in (temp_all, deaths_all):
    d["date"] = pd.to_datetime(d["date"])
    d["year"], d["month"] = d["date"].dt.year, d["date"].dt.month


# ---- 2. Build the département-month panel --------------------
def build_panel(exclude_years=()):
    temp = temp_all[temp_all["year"].between(2000, 2025) & ~temp_all["year"].isin(exclude_years)
                    & temp_all["month"].isin(SUMMER_MONTHS)].copy()
    deaths = deaths_all[deaths_all["year"].between(2000, 2025) & ~deaths_all["year"].isin(exclude_years)]

    # heat-wave days per département-month
    temp["hw_day"] = temp["canicule_official"].fillna(0).eq(1)
    hw = temp.groupby(["dep", "year", "month"])["hw_day"].sum().rename("hw_days").reset_index()

    # average month of each year (population growth and ageing)
    annual = deaths.groupby(["dep", "year"])["deces_total"].sum().rename("annual").reset_index()
    annual["monthly_avg"] = annual["annual"] / 12

    # summer monthly deaths and deviation from the standard summer
    dm = (deaths[deaths["month"].isin(SUMMER_MONTHS)]
          .groupby(["dep", "departement", "year", "month"])["deces_total"].sum().rename("deaths").reset_index())
    p = (dm.merge(hw, on=["dep", "year", "month"])
           .merge(annual[["dep", "year", "monthly_avg"]], on=["dep", "year"]))
    p["ratio"] = p["deaths"] / p["monthly_avg"]
    p["standard_ratio"] = p.groupby(["dep", "month"])["ratio"].transform("mean")
    p["expected"] = p["standard_ratio"] * p["monthly_avg"]
    p["excess"] = p["deaths"] - p["expected"]
    p["dev_pct"] = p["excess"] / p["expected"] * 100

    # heat-wave exposure variables
    p["category"] = np.where(p["hw_days"] >= 1, "Heat wave", "No heat wave")
    p["any_hw"] = (p["hw_days"] >= 1).astype(float)
    p["hw_bin"] = pd.cut(p["hw_days"], bins=BINS, labels=BIN_LABELS)
    for lab in BIN_LABELS[1:]:
        p[f"bin_{lab}"] = (p["hw_bin"] == lab).astype(float)
    p["year_month"] = p["year"].astype(str) + "-" + p["month"].astype(str)
    return p


panels = {key: build_panel(excl) for key, (excl, _) in SAMPLES.items()}


# ---- 3. Group means with 95% CI ------------------------------
def summarise(df):
    n, m, sd = len(df), df["dev_pct"].mean(), df["dev_pct"].std()
    se = sd / np.sqrt(n) if n > 1 else np.nan
    return pd.Series({"n_months": n, "mean_dev_pct": m, "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se,
                      "excess_deaths": df["excess"].sum()})


# ---- 4. Plot helpers -----------------------------------------
YLABEL = "Deviation from standard summer deaths (%)"


def foot(period):
    return ("Heat-wave day: canicule_official = 1. Each month is divided by the average month of its own year "
            "(annual deaths / 12) to account for population growth and ageing;\nstandard summer = usual value of "
            f"that ratio for the same département and calendar month, {period}. n = département-months. "
            "Bars: mean deviation, whiskers: 95% CI.")


def style(ax, ymin, ymax):
    ax.axhline(0, color=INK2, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(min(0, ymin) - 2.5, ymax * 1.18 + 1)


def bar(ax, x, row, color, width):
    m = row["mean_dev_pct"]
    ax.bar(x, m, width=width, color=color, edgecolor="white", linewidth=2)
    ax.errorbar(x, m, yerr=[[m - row["ci_low"]], [row["ci_high"] - m]], color=INK, lw=1.2, capsize=4)
    up = m >= 0
    ax.text(x, (row["ci_high"] if up else row["ci_low"]) + (0.3 if up else -0.3), f"{m:+.1f}%",
            ha="center", va="bottom" if up else "top", color=INK, fontsize=9)


def finish(fig, title, period, name):
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.text(0.01, 0.01, foot(period), fontsize=8, color=INK2)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(name)
    if SAVE:
        os.makedirs(os.path.join(DATA_DIR, "figures"), exist_ok=True)
        fig.savefig(os.path.join(DATA_DIR, "figures", name + ".png"), dpi=200)


# ---- 5. Figures ----------------------------------------------
for key in GRAPH_SAMPLES:
    p, period = panels[key], SAMPLES[key][1]
    overall = p.groupby("category").apply(summarise, include_groups=False)
    by_month = p.groupby(["month", "category"]).apply(summarise, include_groups=False)
    by_bin = p.groupby("hw_bin", observed=False).apply(summarise, include_groups=False)
    suffix = "" if key == "all" else "_without_2003"

    # Figure 1: with vs without a heat-wave day
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2]})
    cats = [("No heat wave", C_NO), ("Heat wave", C_HW)]
    for i, (cat, col) in enumerate(cats):
        bar(ax1, i, overall.loc[cat], col, 0.6)
    ax1.set_xticks([0, 1], [f"{c}\n(n = {int(overall.loc[c, 'n_months']):,})" for c, _ in cats])
    ax1.set_title("A. All summer months (June–September)", loc="left", color=INK, fontsize=11)
    ax1.set_ylabel(YLABEL)
    style(ax1, overall["ci_low"].min(), overall["ci_high"].max())

    def n_of(m, c):
        return int(by_month.loc[(m, c), "n_months"]) if (m, c) in by_month.index else 0

    w = 0.38
    for j, mo in enumerate(SUMMER_MONTHS):
        for k, (cat, col) in enumerate(cats):
            if (mo, cat) in by_month.index:
                bar(ax2, j + (k - 0.5) * w, by_month.loc[(mo, cat)], col, w)
    ax2.set_xticks(range(len(SUMMER_MONTHS)),
                   [f"{MONTH_NAMES[m]}\n(n = {n_of(m, 'No heat wave'):,} / {n_of(m, 'Heat wave')})"
                    for m in SUMMER_MONTHS])
    ax2.set_title("B. By calendar month (n = no heat wave / heat wave)", loc="left", color=INK, fontsize=11)
    ax2.set_ylabel(YLABEL)
    style(ax2, by_month["ci_low"].min(), by_month["ci_high"].max())
    ax2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_NO), plt.Rectangle((0, 0), 1, 1, color=C_HW)],
               labels=["No heat-wave day in the month", "At least one heat-wave day"],
               frameon=False, loc="upper left")
    finish(fig, "Monthly deaths vs a standard summer: months with and without a heat-wave day\n"
                f"France, 95 départements, {period} (adjusted for population growth and ageing)",
           period, f"fig1_heatwave_vs_no_heatwave{suffix}")

    # Figure 2: by number of heat-wave days
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, lab in enumerate(BIN_LABELS):
        bar(ax, i, by_bin.loc[lab], BIN_COLORS[i], 0.65)
    ax.set_xticks(range(len(BIN_LABELS)),
                  [f"{l}\n(n = {int(n):,})" for l, n in zip(BIN_LABELS, by_bin["n_months"])])
    ax.set_xlabel("Official heat-wave days in the month")
    ax.set_ylabel(YLABEL)
    style(ax, by_bin["ci_low"].min(), by_bin["ci_high"].max())
    finish(fig, "Monthly deaths by number of heat-wave days in the month\n"
                f"France, 95 départements, June–September {period} (adjusted for population growth and ageing)",
           period, f"fig2_deviation_by_hw_days{suffix}")

    print(f"\n=== {period} ===")
    print(by_bin[["n_months", "mean_dev_pct", "ci_low", "ci_high"]].round(2))


# ---- 6. OLS with fixed effects and clustered SE --------------
def demean(df, cols, fe, tol=1e-10, max_iter=500):
    """Remove fixed effects by alternating projections (Frisch–Waugh–Lovell)."""
    out = df[cols].astype(float).copy()
    for _ in range(max_iter):
        old = out.copy()
        for g in fe:
            out = out - out.groupby(df[g]).transform("mean")
        if (out - old).abs().to_numpy().max() < tol:
            break
    return out


def ols(df, y, xs, fe=(), cluster="dep"):
    df = df.dropna(subset=[y] + xs).reset_index(drop=True)
    if fe:
        dm = demean(df, [y] + xs, fe)
        Y, X, names = dm[y].to_numpy(), dm[xs].to_numpy(), xs
    else:
        Y = df[y].to_numpy()
        X = np.column_stack([np.ones(len(df)), df[xs].to_numpy()])
        names = ["const"] + xs
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta

    meat = np.zeros((k, k))                     # cluster-robust (CR1) variance
    groups = df[cluster].to_numpy()
    for g in np.unique(groups):
        s = X[groups == g].T @ resid[groups == g]
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    k_eff = k + sum(df[f].nunique() - (1 if i else 0) for i, f in enumerate(fe))
    corr = G / (G - 1) * (n - 1) / max(n - k_eff, 1)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv * corr))

    y_raw = df[y].to_numpy()
    r2 = 1 - (resid ** 2).sum() / ((y_raw - y_raw.mean()) ** 2).sum()
    return {"coef": dict(zip(names, beta)), "se": dict(zip(names, se)), "n": n, "r2": r2}


BIN_VARS = [f"bin_{lab}" for lab in BIN_LABELS[1:]]
FE = ["dep", "year_month"]
specs = [("all", ["any_hw"], ()), ("all", BIN_VARS, ()), ("all", BIN_VARS, FE),
         ("no2003", ["any_hw"], ()), ("no2003", BIN_VARS, ()), ("no2003", BIN_VARS, FE)]
results = [ols(panels[s], "dev_pct", xs, fe) for s, xs, fe in specs]

# ---- 7. Regression table -------------------------------------
ROWS = [("any_hw", "≥ 1 heat-wave day"), ("bin_1–3", "1–3 heat-wave days"), ("bin_4–6", "4–6 heat-wave days"),
        ("bin_7–9", "7–9 heat-wave days"), ("bin_10+", "10+ heat-wave days"), ("const", "Constant")]


def stars(b, se):
    t = abs(b / se)
    return "***" if t > 2.576 else "**" if t > 1.96 else "*" if t > 1.645 else ""


table = []
for var, label in ROWS:
    c_row, s_row = [label], [""]
    for r in results:
        if var in r["coef"]:
            b, se = r["coef"][var], r["se"][var]
            c_row.append(f"{b:.2f}{stars(b, se)}")
            s_row.append(f"({se:.2f})")
        else:
            c_row.append("")
            s_row.append("")
    table += [c_row, s_row]
footer = [["Sample"] + ["2000–2025"] * 3 + ["Excl. 2003"] * 3,
          ["Département FE"] + ["No", "No", "Yes"] * 2,
          ["Year × month FE"] + ["No", "No", "Yes"] * 2,
          ["Observations"] + [f"{r['n']:,}" for r in results],
          ["R²"] + [f"{r['r2']:.3f}" for r in results]]
cols = ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"]
print("\n", pd.DataFrame(table + footer, columns=cols).to_string(index=False))

# draw the table
row_h = 0.32
fig_h = 1.9 + row_h * (len(table) + len(footer) + 2)
fig, ax = plt.subplots(figsize=(11, fig_h))
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, fig_h)
x_label, x_cols = 0.02, np.linspace(0.36, 0.95, 6)

y = fig_h - 0.35
ax.text(x_label, y, "Heat-wave days and deviation of monthly deaths from a standard summer",
        fontsize=13, color=INK, va="top", fontweight="bold")
y -= 0.38
ax.text(x_label, y, "Dependent variable: deviation from standard summer deaths (%), département-month, June–September",
        fontsize=9.5, color=INK2, va="top")
y -= 0.45
ax.plot([0.01, 0.99], [y, y], color=INK, lw=1.2)
y -= 0.12
ax.text(np.mean(x_cols[:3]), y, "All years", ha="center", va="top", fontsize=10, color=INK)
ax.text(np.mean(x_cols[3:]), y, "Without 2003", ha="center", va="top", fontsize=10, color=INK)
y -= 0.28
for xs_ in (x_cols[:3], x_cols[3:]):
    ax.plot([xs_[0] - 0.04, xs_[-1] + 0.04], [y, y], color=INK2, lw=0.6)
y -= 0.06
for x, c in zip(x_cols, cols[1:]):
    ax.text(x, y, c, ha="center", va="top", fontsize=10, color=INK)
y -= 0.32
ax.plot([0.01, 0.99], [y, y], color=INK, lw=0.6)
y -= 0.08
for i, row in enumerate(table):
    is_se = i % 2 == 1
    for j, cell in enumerate(row):
        if j == 0:
            ax.text(x_label, y, cell, ha="left", va="top", fontsize=10, color=INK)
        else:
            ax.text(x_cols[j - 1], y, cell, ha="center", va="top", fontsize=9 if is_se else 10,
                    color=INK2 if is_se else INK)
    y -= row_h * (0.85 if is_se else 0.75)
y -= 0.05
ax.plot([0.01, 0.99], [y, y], color=INK, lw=0.6)
y -= 0.08
for row in footer:
    for j, cell in enumerate(row):
        ax.text(x_label if j == 0 else x_cols[j - 1], y, cell, ha="left" if j == 0 else "center",
                va="top", fontsize=9.5, color=INK)
    y -= row_h
ax.plot([0.01, 0.99], [y + 0.04, y + 0.04], color=INK, lw=1.2)
y -= 0.1
ax.text(x_label, y, "Reference category: months with 0 heat-wave days (canicule_official = 1). Each month is divided by "
                    "the average month of its own year (annual deaths / 12),\nthen compared with the usual value for the "
                    "same département and calendar month. Standard errors clustered by département (95 clusters) in "
                    "parentheses.\n* p < 0.10, ** p < 0.05, *** p < 0.01. With fixed effects, the constant is absorbed "
                    "and R² includes the fixed effects.", fontsize=8, color=INK2, va="top", linespacing=1.5)
if fig.canvas.manager is not None:
    fig.canvas.manager.set_window_title("regression_table_heatwave_days")
if SAVE:
    os.makedirs(os.path.join(DATA_DIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(DATA_DIR, "figures", "regression_table_heatwave_days.png"),
                dpi=200, bbox_inches="tight", facecolor="white")

# ---- 8. Open all windows -------------------------------------
plt.show()
