# OmniSight: Multimodal Photo Search & 3D Vector Explorer

OmniSight is a localized, intelligent media engine that transforms flat image directories into a searchable, semantic knowledge graph. By fusing visual features with automatically enriched metadata (EXIF timeframes, reverse-geocoded spatial addresses, and generated captions), the system allows users to query their personal galleries using complex natural language, visual similarity, or both simultaneously.

---

## 🚀 Key Features

* **Multimodal Embedding Fusion:** Leverages Open-CLIP (ViT-L-14) to encode both textual descriptions and visual properties into a normalized dual-embedding space.
* **Automated AI Captioning:** Integrates BLIP (Image Captioning Large) pipelines to extract descriptions for uncaptioned personal images.
* **Temporal & Spatial Enrichment:** Extracts raw EXIF/HEIC data, translates raw GPS coordinates into human-readable locations via GeoPy, and formats timestamps into conversational descriptors (e.g., "14th January 2026, morning, winter").
* **Scalable Vector Search:** Implements a dynamic FAISS backend that transitions smoothly from flat index profiles (IndexFlatIP) for small datasets to Quantized Inverted File systems (IndexIVFPQ) as your photo library expands.
* **3D Latent Space Analytics:** Compresses high-dimensional vectors via UMAP into interactive 3D graphs powered by Plotly, uncovering geometric image clusters via DBSCAN density modeling.

---

## 🛠️ System Architecture

1. Stage 1 (Metadata Extraction Pipeline): Multi-threaded ingestion (ThreadPoolExecutor) crawls your asset directories, reading file headers, calculating reverse-geocoding coordinates, generating BLIP contextual captions, and preserving local state inside a structured SQLite3 instance.
2. Stage 2 (Vector Generation & Mapping): Iterates through unprocessed DB entries, loading batch images alongside parsed metadata strings, generating weighted multimodel features (30% text metadata context + 70% raw visual feature weights), and appending them into the indexed FAISS cluster map.
3. Stage 3 (Streamlit Frontend App): Provides a reactive web UI splits into dual operational panels: Semantic multi-condition image searches or interactive cluster exploration.

---

## 📦 Project Structure

├── app.py                # Main Streamlit Dashboard Application
├── main.py               # Orchestrator running Stage 1 & Stage 2 pipelines
├── stage_1.py            # SQLite metadata extraction, GPS parsing, & BLIP pipeline
├── stage_2.py            # FAISS vector quantization and feature fusion engine
├── photo_metadata.db     # SQLite local metadata database
├── faiss.index           # Persisted multi-dimensional FAISS index
├── faiss_map.json        # ID-to-filename structural lookups
└── photos/               # Target directory for your image library (.heic, .jpg, .png)

---

## 🔧 Installation & Local Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed. This project natively supports hardware acceleration on Apple Silicon (mps) and CUDA configurations.

### 2. Dependencies
Install the required packages:
pip install streamlit torch open_clip_torch faiss-cpu pillow pillow-heif umap-learn plotly scikit-learn geopy transformers

### 3. Execution Pipeline

First, place your images into the local photos/ folder. Next, parse your data library and construct your index profiles:
python main.py

Launch the interactive Streamlit engine:
streamlit run app.py

---

## 📊 Analytics Deep-Dive

The visualization portal performs a mathematical breakdown of your media profile distribution:
* **Dimensionality Reduction:** Reduces 768-dimension CLIP features down to a compact 3-axis grid using Uniform Manifold Approximation and Projection (UMAP).
* **Semantic Clustering:** Automatically isolates conceptual groups using Density-Based Spatial Clustering of Applications with Noise (DBSCAN), neutralizing anomalous outliers.
* **Topic Modeling:** Extracts structural descriptions of each vector community on-the-fly using TF-IDF keyword weights derived from the cached SQLite metadata table.
