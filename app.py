import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import streamlit as st
import torch
import open_clip
import faiss
import json
import numpy as np
import sqlite3
import time
from PIL import Image
from pathlib import Path
import umap
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


# ================================
# STYLES & UI THEME (JARVIS)
# ================================
def apply_jarvis_theme():
    st.markdown("""
        <style>
        /* Main background and text colors */
        .stApp {
            background-color: #060b14;
            color: #00f2ff;
        }
        
        /* Holographic Glow for headers */
        h1, h2, h3 {
            color: #00f2ff !important;
            text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff;
            font-family: 'Courier New', Courier, monospace;
        }

        /* High-tech Borders for images and buttons */
        .stButton>button {
            border: 2px solid #00f2ff !important;
            background-color: rgba(0, 242, 255, 0.1) !important;
            color: #00f2ff !important;
            border-radius: 0px !important;
            transition: 0.3s;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .stButton>button:hover {
            background-color: rgba(0, 242, 255, 0.3) !important;
            box-shadow: 0 0 15px #00f2ff;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0a111e;
            border-right: 1px solid #00f2ff;
        }

        /* Custom Loading Animation Overlay */
        .jarvis-load {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 80vh;
        }
        
        /* Interactive Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            color: #555;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 242, 255, 0.1) !important;
            border-bottom: 2px solid #00f2ff !important;
            color: #00f2ff !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ================================
# JARVIS BOOT SEQUENCE (Loading)
# ================================
def boot_sequence():
    if 'booted' not in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("""
                <div class="jarvis-load">
                    <img src="https://i.pinimg.com/originals/11/1a/11/111a1170756708466e33e11f77041183.gif" width="300">
                    <h2 style="margin-top: 20px;">INITIALIZING SYSTEMS...</h2>
                    <p style="font-family: monospace; color: #00f2ff;">Accessing Neural Network | Loading CLIP Weights | Calibrating Sensors</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Simulate high-tech loading progress
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        
        placeholder.empty()
        st.session_state['booted'] = True
        st.toast("Welcome back, Sir. Systems are online.", icon="🦾")

# ================================
# CORE LOGIC
# ================================
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except:
    pass

FAISS_INDEX_FILE = "faiss.index"
FAISS_MAP_FILE = "faiss_map.json"
IMAGE_FOLDER = "photos"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Systems Online. Neural processing units redirected to: {device}")

@st.cache_resource
def load_models():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer

@st.cache_resource
def load_faiss():
    index = faiss.read_index(FAISS_INDEX_FILE)
    if hasattr(index, "nprobe"): index.nprobe = 10
    with open(FAISS_MAP_FILE, "r") as f:
        id_map = json.load(f)
    return index, id_map

@st.cache_resource
def load_all_vectors():
    index, _ = load_faiss()
    return np.array([index.reconstruct(i) for i in range(index.ntotal)]).astype("float32")

# ================================
# MAIN UI
# ================================
apply_jarvis_theme()
boot_sequence()

model, preprocess, tokenizer = load_models()
index, id_map = load_faiss()

tab1, tab2 = st.tabs(["[ 🔍 NEURAL SEARCH ]", "[ 📊 VISUALIZATION ]"])

# --- SEARCH TAB ---
with tab1:
    st.title("🦾 STARK INDUSTRIES: PHOTO ARCHIVE")
    st.write("---")
    
    with st.container():
        c1, c2 = st.columns([2, 1])
        with c1:
            query = st.text_input("QUERY INPUT", placeholder="Type a concept (e.g., 'Target in Goa' or 'Tech Gear')")
        with c2:
            uploaded_file = st.file_uploader("VISUAL UPLOAD", type=["jpg", "png", "jpeg"])

    top_k = st.sidebar.slider("ACCURACY DEPTH (TOP K)", 1, 100, 50)
    search_clicked = st.button("RUN SCAN")

    if search_clicked:
        with st.spinner("SCANNING DATABASE..."):
            # Embedding logic (Same as yours)
            text_emb = None
            if query:
                tokens = tokenizer([query]).to(device)
                with torch.no_grad():
                    text_emb = model.encode_text(tokens)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)
                text_emb = text_emb.cpu().numpy().astype("float32")

            image_emb = None
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")
                image_emb = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_emb = model.encode_image(image_emb)
                image_emb /= image_emb.norm(dim=-1, keepdim=True)
                image_emb = image_emb.cpu().numpy().astype("float32")

            # Fusion
            if text_emb is not None and image_emb is not None:
                final_emb = 0.3 * text_emb + 0.7 * image_emb
            else:
                final_emb = text_emb if text_emb is not None else image_emb

            if final_emb is not None:
                scores, indices = index.search(final_emb, top_k)
                
                cols = st.columns(3)
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1: continue
                    fname = id_map[str(idx)]
                    path = Path(IMAGE_FOLDER) / fname
                    with cols[i % 3]:
                        st.markdown(f'<div style="border: 1px solid #00f2ff; padding: 5px;">', unsafe_allow_html=True)
                        st.image(str(path), use_container_width=True)
                        st.caption(f"MATCH: {score:.4f} | ID: {fname}")
                        st.markdown('</div>', unsafe_allow_html=True)

# --- VISUALIZATION TAB ---
with tab2:
    st.title("📊 CLUSTER TOPOLOGY")
    
    with st.status("Generating 3D Neural Map...", expanded=True):
        vectors = load_all_vectors()
        reducer = umap.UMAP(n_components=3, random_state=42)
        vectors_3d = reducer.fit_transform(vectors)
        
        clustering = DBSCAN(eps=0.5, min_samples=5)
        labels = clustering.fit_predict(vectors_3d)
    
    # 3D Plot with Jarvis Colors
    fig = px.scatter_3d(
        x=vectors_3d[:,0], y=vectors_3d[:,1], z=vectors_3d[:,2],
        color=labels.astype(str),
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.cyan # Lowercase 'c'
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)', # Makes plot background transparent
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    # Explorer Section
    st.write("---")
    st.subheader("📁 SUB-SYSTEM EXPLORER")
    
    c1, c2 = st.columns(2)
    with c1:
        valid_clusters = [c for c in set(labels) if c != -1]
        sel_cluster = st.selectbox("CHOOSE SECTOR", valid_clusters)
    
    # Show Cluster Images
    cluster_cols = st.columns(5)
    ci = 0
    for i, label in enumerate(labels):
        if label == sel_cluster:
            fname = id_map[str(i)]
            path = Path(IMAGE_FOLDER) / fname
            with cluster_cols[ci % 5]:
                st.image(str(path), use_container_width=True)
            ci += 1