import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import open_clip
import faiss
import json
import numpy as np
import sqlite3

from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

# Visualization
import umap
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

register_heif_opener()

FAISS_INDEX_FILE = "faiss.index"
FAISS_MAP_FILE = "faiss_map.json"
IMAGE_FOLDER = "photos"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer

@st.cache_resource
def load_faiss():
    index = faiss.read_index(FAISS_INDEX_FILE)

    if hasattr(index, "nprobe"):
        index.nprobe = 10

    with open(FAISS_MAP_FILE, "r") as f:
        id_map = json.load(f)

    return index, id_map

@st.cache_resource
def load_all_vectors():
    index = faiss.read_index(FAISS_INDEX_FILE)
    return index.reconstruct_n(0, index.ntotal)

model, preprocess, tokenizer = load_models()
index, id_map = load_faiss()

# ================================
# EMBEDDINGS
# ================================
def text_to_embedding(query):
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype("float32")

def image_to_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype("float32")

def combine_embeddings(text_emb, image_emb):
    if text_emb is not None and image_emb is not None:
        return 0.3 * text_emb + 0.7 * image_emb
    elif text_emb is not None:
        return text_emb
    elif image_emb is not None:
        return image_emb
    return None

# ================================
# SEARCH
# ================================
def search(query_embedding, top_k=5):
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        fname = id_map[str(idx)]
        path = Path(IMAGE_FOLDER) / fname

        results.append({
            "file_name": fname,
            "score": float(score),
            "path": str(path)
        })

    return results

# ================================
# DB HELPERS
# ================================
def load_caption(file_name):
    conn = sqlite3.connect("photo_metadata.db")
    cursor = conn.cursor()

    cursor.execute("SELECT caption FROM metadata WHERE file_path = ?", (file_name,))
    row = cursor.fetchone()
    conn.close()

    return row[0] if row else ""

# ================================
# TF-IDF CLUSTER MEANING
# ================================
def get_cluster_descriptions(labels, id_map):
    cluster_text = {}

    for cid in set(labels):
        if cid == -1:
            continue

        captions = []
        for i, l in enumerate(labels):
            if l == cid:
                fname = id_map[str(i)]
                cap = load_caption(fname)
                if cap:
                    captions.append(cap)

        if not captions:
            continue

        vec = TfidfVectorizer(stop_words="english", max_features=5)
        vec.fit(captions)

        cluster_text[cid] = ", ".join(vec.get_feature_names_out())

    return cluster_text

# ================================
# UI
# ================================
st.set_page_config(page_title="AI Photo Search", layout="wide")

tab1, tab2 = st.tabs(["🔍 Search", "📊 Visualization"])

# ================================
# TAB 1: SEARCH
# ================================
with tab1:
    st.title("🤖 Multimodal Photo Search")

    query = st.text_input("💬 Describe what you're looking for")

    uploaded_file = st.file_uploader(
        "📎 Upload image (optional)",
        type=["jpg", "png", "jpeg"]
    )

    col1, col2 = st.columns([1, 5])

    with col1:
        top_k = st.slider("Top K", 1, 100, 50)

    with col2:
        search_clicked = st.button("🚀 Search")

    if search_clicked:
        text_emb = text_to_embedding(query) if query else None

        image_emb = None
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=200)
            image_emb = image_to_embedding(image)

        final_embedding = combine_embeddings(text_emb, image_emb)

        if final_embedding is None:
            st.warning("Enter text or upload image")
        else:
            results = search(final_embedding, top_k)

            cols = st.columns(3)
            for i, r in enumerate(results):
                with cols[i % 3]:
                    st.image(r["path"], use_container_width=True)
                    st.caption(r["file_name"])
                    st.write(f"{r['score']:.4f}")

# ================================
# TAB 2: VISUALIZATION
# ================================
with tab2:
    st.title("📊 3D Embedding Visualization")

    if st.button("Generate Visualization"):

        vectors = load_all_vectors()


        # UMAP
        reducer = umap.UMAP(n_components=3, random_state=42)
        vectors_3d = reducer.fit_transform(vectors)

        # DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=5)
        labels = clustering.fit_predict(vectors_3d)

        # Cluster meaning
        cluster_text = get_cluster_descriptions(labels, id_map)

        x, y, z = vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2]

        hover = []
        for i in range(len(x)):
            fname = id_map[str(i)]
            cap = load_caption(fname)
            hover.append(f"{fname}<br>{cap}")

        fig = px.scatter_3d(x=x, y=y, z=z, color=labels.astype(str), hover_name=hover)

        # 🔥 REMOVE AXES
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=30)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Cluster insights
        st.subheader("🧠 Cluster Insights")
        for cid, desc in cluster_text.items():
            st.write(f"Cluster {cid}: {desc}")

        # Cluster explorer
        st.subheader("🖼️ Explore Cluster")

        valid_clusters = [c for c in set(labels) if c != -1]

        selected_cluster = st.selectbox("Select Cluster", valid_clusters)

        cols = st.columns(5)
        idx = 0

        for i, label in enumerate(labels):
            if label == selected_cluster:
                fname = id_map[str(i)]
                path = Path(IMAGE_FOLDER) / fname

                with cols[idx % 5]:
                    st.image(str(path), use_container_width=True)
                    st.caption(fname)

                idx += 1

        # 🔥 CLICK SUBSTITUTE VIEWER
        st.subheader("🖱️ Select Image")

        options = [f"{i} | {id_map[str(i)]}" for i in range(len(x))]
        options = options[:500]  # prevent lag

        selected = st.selectbox("Select Image", options)

        if selected:
            idx = int(selected.split(" | ")[0])
            fname = id_map[str(idx)]
            path = Path(IMAGE_FOLDER) / fname

            st.image(str(path), width=400)
            st.caption(fname)

            cap = load_caption(fname)
            st.write("🧠 Caption:", cap)