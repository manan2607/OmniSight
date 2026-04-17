import numpy as np
import faiss
import json
import sqlite3

from sklearn.cluster import KMeans
import umap
import plotly.express as px
from collections import Counter

# ================================
# LOAD FAISS + MAPPING
# ================================
index = faiss.read_index("faiss.index")

with open("faiss_map.json", "r") as f:
    id_map = json.load(f)

# ================================
# GET ALL VECTORS FROM FAISS
# ================================
vectors = index.reconstruct_n(0, index.ntotal)

# ================================
# REDUCE TO 3D (UMAP)
# ================================
reducer = umap.UMAP(n_components=3, random_state=42)
vectors_3d = reducer.fit_transform(vectors)

# ================================
# CLUSTERING
# ================================
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(vectors_3d)

# ================================
# LOAD CAPTIONS FROM DB
# ================================
def load_caption(file_name):
    conn = sqlite3.connect("photo_metadata.db")
    cursor = conn.cursor()

    cursor.execute("SELECT caption FROM metadata WHERE file_path = ?", (file_name,))
    row = cursor.fetchone()
    conn.close()

    return row[0] if row else ""

# ================================
# GET COMMON WORDS PER CLUSTER
# ================================
cluster_text = {}

for i in range(k):
    words = []

    for idx, label in enumerate(labels):
        if label == i:
            fname = id_map[str(idx)]
            caption = load_caption(fname)

            words.extend(caption.lower().split())

    common = Counter(words).most_common(5)
    cluster_text[i] = ", ".join([w[0] for w in common])

# ================================
# PREPARE DATA FOR PLOT
# ================================
x, y, z = vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2]

hover_text = []
for i in range(len(x)):
    fname = id_map[str(i)]
    caption = load_caption(fname)
    cluster = labels[i]

    hover_text.append(f"{fname}<br>{caption}<br>Cluster: {cluster}")

# ================================
# 3D PLOT
# ================================
fig = px.scatter_3d(
    x=x, y=y, z=z,
    color=labels,
    hover_name=hover_text,
    title="Image Embedding Clusters"
)

fig.show()