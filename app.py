import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import open_clip
import faiss
import json
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()


FAISS_INDEX_FILE = "faiss.index"
FAISS_MAP_FILE = "faiss_map.json"
IMAGE_FOLDER = "photos"

device = "mps" if torch.backends.mps.is_available() else "cpu"


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

model, preprocess, tokenizer = load_models()
index, id_map = load_faiss()


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
        return 0.5 * text_emb + 0.5 * image_emb
    elif text_emb is not None:
        return text_emb
    elif image_emb is not None:
        return image_emb
    else:
        return None


def search(query_embedding, top_k=5):
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        file_name = id_map[str(idx)]
        img_path = Path(IMAGE_FOLDER) / file_name

        results.append({
            "file_name": file_name,
            "score": float(score),
            "path": str(img_path)
        })

    return results


st.set_page_config(page_title="AI Photo Search", layout="wide")

st.title("🤖 Multimodal Photo Search")
st.caption("Text + Image together (like ChatGPT)")


query = st.text_input("💬 Describe what you're looking for")

uploaded_file = st.file_uploader(
    "📎 (Optional) Upload an image to refine search",
    type=["jpg", "png", "jpeg"]
)

col1, col2 = st.columns([1, 5])

with col1:
    top_k = st.slider("Top K", 1, 10, 5)

with col2:
    search_clicked = st.button("🚀 Search")


if search_clicked:

    text_emb = None
    image_emb = None

    # TEXT
    if query:
        text_emb = text_to_embedding(query)

    # IMAGE
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Query Image", width=250)

        image_emb = image_to_embedding(image)

    # COMBINE
    final_embedding = combine_embeddings(text_emb, image_emb)

    if final_embedding is None:
        st.warning("Please enter text or upload an image")
    else:
        results = search(final_embedding, top_k)

        st.subheader("🔍 Results")

        cols = st.columns(3)

        for i, r in enumerate(results):
            with cols[i % 3]:
                st.image(r["path"], use_container_width=True)
                st.caption(r["file_name"])
                st.write(f"Score: {r['score']:.4f}")