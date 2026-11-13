"""
EcoCor Network – Network Only
Same layout as iter23, but without logo, heading, description, and legend.
Network is centered in the full DIN A0 landscape figure.
Exports: network_only.svg, network_only.png, network_only.pdf
"""

import os
import networkx as nx
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from fa2_modified import ForceAtlas2
from adjustText import adjust_text
from networkx.algorithms.community import louvain_communities

for fname in ["Rubik-Regular.ttf", "Rubik-Bold.ttf"]:
    path = f"/opt/anaconda3/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/{fname}"
    if os.path.exists(path):
        fm.fontManager.addfont(path)

WORK_DIR  = ""
GEXF_PATH = f"{WORK_DIR}/de_corpus_w3.gexf"
OUT_SVG   = f"{WORK_DIR}/network_only.svg"
OUT_PNG   = f"{WORK_DIR}/network_only.png"
OUT_PDF   = f"{WORK_DIR}/network_only.pdf"

# ── Colour palette ───────────────────────────────────────────────────────────
CLOUD    = "#F4F7F5"
GRAPHITE = "#2A2D34"
FONT     = "Rubik"

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

W_MM, H_MM = 1189, 841
DPI_CALC   = 300
W_IN = W_MM / 25.4
H_IN = H_MM / 25.4
W_PX = W_IN * DPI_CALC
H_PX = H_IN * DPI_CALC
DPI  = 120   # render DPI

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data …")
G_full   = nx.read_gexf(GEXF_PATH)
G_layout = G_full.copy()

for _g in [G_full, G_layout]:
    for n, d in _g.nodes(data=True):
        d["degree"]      = int(d.get("degree", _g.degree(n)))
        d["betweenness"] = float(d.get("betweenness_centrality", 0))
        d["entity_type"] = d.get("entity_type", "Animal")

G_layout.remove_edges_from([
    (u, v) for u, v, d in G_layout.edges(data=True)
    if float(d.get("text_count", 0)) < 3
])

# ── 2. Community detection ────────────────────────────────────────────────────
print("Louvain community detection …")
comms = louvain_communities(G_full, seed=42, weight="text_count")
comms_sorted = sorted(comms, key=len, reverse=True)
print(f"  {len(comms_sorted)} communities")

node_comm = {}
for i, comm in enumerate(comms_sorted):
    label = i if i <= 3 else "small"
    for n in comm:
        node_comm[n] = label
for n, d in G_full.nodes(data=True):
    d["comm_label"] = node_comm.get(n, "small")

# ── 3. Core / fringe ──────────────────────────────────────────────────────────
components   = sorted(nx.connected_components(G_layout), key=len, reverse=True)
core_nodes   = set(components[0])
fringe_nodes = [n for c in components[1:] for n in c]
G_core       = G_layout.subgraph(core_nodes).copy()
print(f"Core: {len(core_nodes)}, fringe: {len(fringe_nodes)}")

# ── 4. ForceAtlas2 ────────────────────────────────────────────────────────────
print("Layout …")
fa2 = ForceAtlas2(
    outboundAttractionDistribution=True, edgeWeightInfluence=1.5,
    jitterTolerance=0.5, barnesHutOptimize=True, barnesHutTheta=1.2,
    scalingRatio=8.0, strongGravityMode=False, gravity=0.3, verbose=True,
)
core_pos = fa2.forceatlas2_networkx_layout(G_core, pos=None, iterations=1000)
print("done.")

# ── 5. Normalise core layout ─────────────────────────────────────────────────
CORE_FILL = 0.76
coords = np.array(list(core_pos.values()))
x_lo, x_hi = np.percentile(coords[:,0], 1.5), np.percentile(coords[:,0], 98.5)
y_lo, y_hi = np.percentile(coords[:,1], 1.5), np.percentile(coords[:,1], 98.5)
cx_raw, cy_raw = (x_lo+x_hi)/2, (y_lo+y_hi)/2
target_px = CORE_FILL * min(W_PX, H_PX)
scale = min(
    target_px / ((x_hi-x_lo) * W_PX),
    target_px / ((y_hi-y_lo) * H_PX),
)

# Network centered in full figure
RING_GAP_PX  = 60
RING_STAGGER = 80
R_base       = 4400
R_outer_px   = R_base + RING_STAGGER   # 4480 px

CENTER_X = 0.5
CENTER_Y = 0.5

positions = {
    n: (CENTER_X + (x-cx_raw)*scale, CENTER_Y + (y-cy_raw)*scale)
    for n, (x, y) in core_pos.items()
}

# ── 6. Clamp outlier positions ───────────────────────────────────────────────
R_core_max = R_base - RING_GAP_PX

outliers = 0
for n in list(core_nodes):
    x, y = positions[n]
    dx, dy = (x-CENTER_X)*W_PX, (y-CENTER_Y)*H_PX
    dist = np.sqrt(dx**2 + dy**2)
    if dist > R_core_max:
        f = R_core_max / dist
        positions[n] = (CENTER_X + dx*f/W_PX, CENTER_Y + dy*f/H_PX)
        outliers += 1
print(f"Outliers clamped: {outliers}")

# ── 7. Fringe: two staggered rings ───────────────────────────────────────────
fringe_sorted = sorted(fringe_nodes,
    key=lambda n: G_full.nodes[n]["betweenness"], reverse=True)
N_fringe = len(fringe_sorted)
N_inner  = (N_fringe + 1) // 2
N_outer  = N_fringe - N_inner

rng_gen = np.random.default_rng(42)
angles_inner = np.linspace(0, 2*np.pi, N_inner, endpoint=False)
angle_j_i    = rng_gen.uniform(-np.pi/N_inner*0.6, np.pi/N_inner*0.6, N_inner)
angles_outer = np.linspace(0, 2*np.pi, N_outer, endpoint=False) + np.pi/N_outer
angle_j_o    = rng_gen.uniform(-np.pi/N_outer*0.6, np.pi/N_outer*0.6, N_outer)
R_inner = R_base

fringe_angles = {}
for idx, n in enumerate(fringe_sorted):
    if idx % 2 == 0:
        k = idx // 2
        a, r = angles_inner[k] + angle_j_i[k], R_inner
    else:
        k = idx // 2
        a, r = angles_outer[k] + angle_j_o[k], R_outer_px
    fringe_angles[n] = a
    positions[n] = (
        np.clip(CENTER_X + r*np.cos(a)/W_PX, 0.005, 0.992),
        np.clip(CENTER_Y + r*np.sin(a)/H_PX, 0.008, 0.992),
    )

# ── 8. Visual mapping ────────────────────────────────────────────────────────
node_list = list(G_full.nodes())
node_idx  = {n: i for i, n in enumerate(node_list)}

def comm_color(n, color_type="face"):
    cl = G_full.nodes[n].get("comm_label", "small")
    return COMM_COLORS[cl] if color_type == "face" else COMM_EDGE_COLORS[cl]

node_colors      = [comm_color(n, "face") for n in node_list]
node_edge_colors = [comm_color(n, "edge") for n in node_list]

bw_vals = np.array([G_full.nodes[n]["betweenness"] for n in node_list])
bw_log  = np.log1p(bw_vals)
bw_norm = bw_log / bw_log.max() if bw_log.max() > 0 else bw_log

SIZE_MIN, SIZE_MAX = 60, 3500
node_sizes = SIZE_MIN + bw_norm * (SIZE_MAX - SIZE_MIN)

FONT_MIN, FONT_MAX = 6.5, 34
font_sizes_arr = FONT_MIN + bw_norm * (FONT_MAX - FONT_MIN)

def node_r_y(sz): return np.sqrt(sz/np.pi) / 72.0 / H_IN

def get_edge_style(u, v, w):
    cu = G_full.nodes[u].get("comm_label", "small")
    cv = G_full.nodes[v].get("comm_label", "small")
    same = (cu == cv)
    color = COMM_COLORS[cu] if same else GRAPHITE
    alpha = float(np.clip(0.05 + 0.30*np.log1p(w)/np.log1p(50), 0.05, 0.55)) if same \
            else float(np.clip(0.02 + 0.06*np.log1p(w)/np.log1p(50), 0.02, 0.10))
    width = float(np.clip(0.2 + 0.5*np.log1p(w)/np.log1p(50), 0.2, 3.0))
    return color, alpha, width

# ── 9. Draw ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
ax.set_facecolor(CLOUD)
fig.patch.set_facecolor(CLOUD)
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

print("Edges …")
cross_edges, same_edges = {}, {}
for u, v, d in G_full.edges(data=True):
    w  = float(d.get("text_count", 1))
    cu = G_full.nodes[u].get("comm_label", "small")
    cv = G_full.nodes[v].get("comm_label", "small")
    color, alpha, width = get_edge_style(u, v, w)
    bucket = (color, round(alpha, 2), round(width, 1))
    (same_edges if cu == cv else cross_edges).setdefault(bucket, []).append((u, v))

for (color, alpha, width), elist in cross_edges.items():
    nx.draw_networkx_edges(G_full, positions, edgelist=elist, ax=ax,
        width=width, edge_color=color, alpha=alpha)
for (color, alpha, width), elist in same_edges.items():
    nx.draw_networkx_edges(G_full, positions, edgelist=elist, ax=ax,
        width=width, edge_color=color, alpha=alpha)

print("Nodes …")
node_scatter = nx.draw_networkx_nodes(G_full, positions, ax=ax,
    nodelist=node_list, node_size=node_sizes,
    node_color=node_colors, linewidths=0.6, edgecolors=node_edge_colors)

# ── 10. Labels ───────────────────────────────────────────────────────────────
print("Labels …")
LABEL_BBOX = dict(boxstyle="round,pad=0.10", facecolor="white",
                  edgecolor="none", alpha=0.70)
all_texts = []

for n in core_nodes:
    x, y  = positions[n]
    i     = node_idx[n]
    r_off = node_r_y(node_sizes[i]) * 1.2 + 0.003
    t = ax.text(x, y + r_off, G_full.nodes[n].get("name", n),
        fontsize=font_sizes_arr[i], ha="center", va="bottom",
        color=GRAPHITE, fontfamily=FONT,
        fontweight="bold" if bw_norm[i] > 0.3 else "normal",
        bbox=LABEL_BBOX, clip_on=True, zorder=5)
    all_texts.append(t)

for n in fringe_nodes:
    x, y  = positions[n]
    i     = node_idx[n]
    a     = fringe_angles[n]
    r_off = node_r_y(node_sizes[i]) * 1.3 + 0.004
    cos_a, sin_a = np.cos(a), np.sin(a)
    ha = "left"   if cos_a >  0.25 else ("right" if cos_a < -0.25 else "center")
    va = "bottom" if sin_a >  0.20 else ("top"   if sin_a < -0.20 else "center")
    t = ax.text(
        x + r_off * cos_a * (H_IN/W_IN), y + r_off * sin_a,
        G_full.nodes[n].get("name", n),
        fontsize=font_sizes_arr[i], ha=ha, va=va,
        color=GRAPHITE, fontfamily=FONT, fontweight="normal",
        bbox=LABEL_BBOX, clip_on=True, zorder=5)
    all_texts.append(t)

print(f"adjustText on {len(all_texts)} labels …")
adjust_text(all_texts, ax=ax, add_objects=[node_scatter],
    expand=(1.12, 1.25), force_text=(0.35, 0.5),
    force_objects=(0.25, 0.35), lim=200,
    only_move={"objects": "y", "text": "xy", "explode": "xy"})
print("adjustText done.")

# ── 11. Export ───────────────────────────────────────────────────────────────
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

META = {
    "Title":   "EcoCor – German Corpus – Animal and Plant Co-occurrence Network",
    "Author":  "EcoCor Project / DH Hackathon Potsdam 2026",
    "Subject": "Network visualization, DIN A0 landscape",
}

print("Saving SVG …")
fig.savefig(OUT_SVG, format="svg", dpi=300, bbox_inches=None, facecolor=CLOUD)
print("Saving PNG …")
fig.savefig(OUT_PNG, format="png", dpi=300, bbox_inches=None, facecolor=CLOUD)
print("Saving PDF …")
fig.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches=None, facecolor=CLOUD,
            metadata=META)
plt.close()
print("Done!")
