"""
EcoCor – Emily Brontë, Wuthering Heights (1847)
Animal–Plant co-occurrence network (w=1).
Core component only, no paratexts.
"""

import os, networkx as nx, numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from adjustText import adjust_text
from networkx.algorithms.community import louvain_communities

# ── Font ──────────────────────────────────────────────────────────────────────
FONT = "Rubik"
for fname in ["Rubik-Regular.ttf", "Rubik-Bold.ttf"]:
    path = f"/opt/anaconda3/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/{fname}"
    if os.path.exists(path):
        fm.fontManager.addfont(path)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = ""
GEXF    = f"{BASE}/output_en/per_text/en_1847_Bronte_Wuthering-Heights_w1.gexf"
OUT_DIR = f"{BASE}/Viz"
STEM    = f"{OUT_DIR}/en_1847_Bronte_Wuthering-Heights"

# ── Colors ────────────────────────────────────────────────────────────────────
CLOUD    = "#F4F7F5"
GRAPHITE = "#2A2D34"

COMM_COLORS = {
    0: "#3F7F4C",
    1: "#4A90E2",
    2: "#F47A3C",
    3: "#E3B623",
    "small": "#5B3FA3",
}
COMM_EDGE_COLORS = {
    0: "#6AAF78",
    1: "#7AB5EE",
    2: "#F9A47A",
    3: "#EDD06A",
    "small": "#8B72CC",
}

# ── Load & annotate ───────────────────────────────────────────────────────────
G_full = nx.read_gexf(GEXF)
for n, d in G_full.nodes(data=True):
    d["freq"] = float(d.get("overall_frequency", 1))

# ── Core component only ───────────────────────────────────────────────────────
comps = sorted(nx.connected_components(G_full), key=len, reverse=True)
G     = G_full.subgraph(comps[0]).copy()
print(f"Core: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Community detection ───────────────────────────────────────────────────────
comms        = louvain_communities(G, seed=42, weight="weight")
comms_sorted = sorted(comms, key=len, reverse=True)
print(f"  {len(comms_sorted)} communities: {[len(c) for c in comms_sorted]}")

node_comm = {}
for i, comm in enumerate(comms_sorted):
    label = i if i <= 3 else "small"
    for n in comm:
        node_comm[n] = label

# ── Layout ────────────────────────────────────────────────────────────────────
raw_pos = nx.kamada_kawai_layout(G, weight="weight")

arr   = np.array(list(raw_pos.values()))
c_arr = arr.mean(axis=0)
span  = max(arr[:,0].ptp(), arr[:,1].ptp())
SCALE = 1.30 / span if span > 0 else 1.0
positions = {n: ((x - c_arr[0]) * SCALE, (y - c_arr[1]) * SCALE)
             for n, (x, y) in raw_pos.items()}

# Figure: squarer (12×11) → naturally more top/bottom breathing room,
# less left/right relative to the old 14×10.
W_IN, H_IN = 12.0, 11.0
LABEL_PAD  = 0.10                          # tighter left/right
X_LIM      = (SCALE * span / 2) + LABEL_PAD
Y_LIM      = X_LIM * H_IN / W_IN          # isotropic base
PPDU       = W_IN * 72 / (2 * X_LIM)      # points per data unit

# ── Visual mapping ────────────────────────────────────────────────────────────
node_list = list(G.nodes())
node_idx  = {n: i for i, n in enumerate(node_list)}

node_fc = [COMM_COLORS[node_comm.get(n, "small")]      for n in node_list]
node_ec = [COMM_EDGE_COLORS[node_comm.get(n, "small")] for n in node_list]

freq      = np.array([G.nodes[n]["freq"] for n in node_list])
freq_norm = freq / freq.max() if freq.max() > 0 else freq

SIZE_MIN, SIZE_MAX = 80, 2400
sizes = SIZE_MIN + freq_norm * (SIZE_MAX - SIZE_MIN)

FS_MIN, FS_MAX = 11.0, 26.0
fss = FS_MIN + freq_norm * (FS_MAX - FS_MIN)

def node_r_data(sz):
    return np.sqrt(sz / np.pi) / PPDU

# ── Draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=150)
ax.set_facecolor(CLOUD)
fig.patch.set_facecolor(CLOUD)
ax.set_axis_off()

# Edges – same-community colored, cross-community subdued
w_max = max((float(d.get("weight", 1.0)) for u, v, d in G.edges(data=True)), default=1.0)
for u, v, d in G.edges(data=True):
    w    = float(d.get("weight", 1.0))
    cu   = node_comm.get(u, "small")
    cv   = node_comm.get(v, "small")
    same = (cu == cv)
    color = COMM_COLORS[cu] if same else GRAPHITE
    alpha = 0.38 if same else 0.12
    lw    = float(0.5 + (w / w_max) * 5.5)
    xu, yu = positions[u]
    xv, yv = positions[v]
    ax.plot([xu, xv], [yu, yv],
            color=color, alpha=alpha, lw=lw, zorder=1,
            solid_capstyle="round")

# Nodes
sc = nx.draw_networkx_nodes(G, positions, ax=ax,
    nodelist=node_list, node_size=sizes,
    node_color=node_fc, linewidths=0.8, edgecolors=node_ec)

# Asymmetric y: labels grow upward, so add extra room at top, trim bottom
ax.set_xlim(-X_LIM, X_LIM)
ax.set_ylim(-Y_LIM + 0.06, Y_LIM + 0.14)

# ── Labels ────────────────────────────────────────────────────────────────────
LBBOX = dict(boxstyle="round,pad=0.10",
             facecolor="white", edgecolor="none", alpha=0.72)

texts = []
for n in node_list:
    x, y  = positions[n]
    i     = node_idx[n]
    name  = G.nodes[n].get("name", n)
    r_off = node_r_data(sizes[i]) * 1.05 + 0.008
    t = ax.text(x, y + r_off, name,
                fontsize=fss[i], ha="center", va="bottom",
                color=GRAPHITE, fontfamily=FONT,
                fontweight="bold" if freq_norm[i] > 0.25 else "normal",
                bbox=LBBOX, clip_on=True, zorder=5)
    texts.append(t)

print(f"adjustText on {len(texts)} labels …")
adjust_text(texts, ax=ax, add_objects=[sc],
            expand=(1.12, 1.25),
            force_text=(0.40, 0.55), force_objects=(0.25, 0.35),
            lim=150,
            only_move={"objects": "y", "text": "xy", "explode": "xy"})
print("adjustText done.")

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

# ── Export ────────────────────────────────────────────────────────────────────
META = {
    "Title":   "EcoCor – Wuthering Heights Animal–Plant Network",
    "Author":  "EcoCor Project / DH Hackathon Potsdam 2026",
    "Subject": "Co-occurrence network, core component, w=1",
}
DPI_OUT = 300

print("Saving SVG …")
fig.savefig(f"{STEM}.svg", format="svg", dpi=DPI_OUT, facecolor=CLOUD)
print("Saving PNG …")
fig.savefig(f"{STEM}.png", format="png", dpi=DPI_OUT, facecolor=CLOUD)
print("Saving PDF …")
fig.savefig(f"{STEM}.pdf", format="pdf", dpi=DPI_OUT, facecolor=CLOUD, metadata=META)
plt.close()
print("Done!")
