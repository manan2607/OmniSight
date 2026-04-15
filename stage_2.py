import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sqlite3
import torch
import open_clip
import numpy as np
import faiss
import json
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()


# Global Config
device = "mps" if torch.backends.mps.is_available() else "cpu"
FAISS_INDEX_FILE = "faiss.index"
FAISS_MAP_FILE = "faiss_map.json"


# CLIP Config
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="laion2b_s32b_b82k"
)

tokenizer = open_clip.get_tokenizer("ViT-L-14")

model = model.to(device)
model.eval()



def metadata_to_text(meta):
    file_name =str(Path(meta.get('file_name','')).name)
    date_time = str(meta.get('date_time', ''))
    address = str(meta.get('address', ''))
    caption = str(meta.get('caption', ''))

    if " " in date_time:
        date, time = date_time.split(" ", 1)
    else:
        date, time = date_time, ""

    parts = [
        caption,
        f"location {address}" if address else "",
        f"date {date}" if date else "",
        f"time {time}" if time else "",
        f"{file_name}",
        "photo"
    ]

    return ", ".join([p for p in parts if p])



def load_data():
    conn = sqlite3.connect('photo_metadata.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM metadata WHERE processed = ? LIMIT ?",
        (0, 50)
    )

    rows = cursor.fetchall()
    conn.close()

    data = []
    for row in rows:
        id, file_name, date_time, location, caption, _ = row

        data.append({
            "id": id,
            "metadata": {
                "file_name": file_name,
                "date_time": date_time,
                "address": location,
                "caption":caption
            }
        })

    return data



def process_batch(batch):
    images = []
    texts = []
    file_names = []
    metadata_list = []

    for item in batch:
        file_name = Path(item['metadata']['file_name'])
        img_path = f"photos/{file_name}"

        try:
            image = Image.open(img_path)

            if not isinstance(image, Image.Image):
                print(f"Invalid image type: {file_name}")
                continue

            image = image.convert("RGB")

        except Exception as e:
            print(f"Error loading: {file_name}, {e}")
            continue

        images.append(preprocess(image).unsqueeze(0))
        texts.append(metadata_to_text(item["metadata"]))
        file_names.append(file_name)
        metadata_list.append(item["metadata"])

    if len(images) == 0:
        return [], [], np.array([])

    images = torch.cat(images).to(device)
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Fusion
    combined = 0.5 * image_features + 0.5 * text_features

    return file_names, metadata_list, combined.cpu().numpy()



def load_mapping():
    if os.path.exists(FAISS_MAP_FILE):
        with open(FAISS_MAP_FILE, "r") as f:
            return json.load(f)
    return {}

def save_mapping(mapping):
    with open(FAISS_MAP_FILE, "w") as f:
        json.dump(mapping, f)



def get_faiss_index(dim, embeddings=None):
    n_vectors = 0 if embeddings is None else embeddings.shape[0]

    # SMALL DATA → use simple index
    if n_vectors < 100:
        print("⚡ Using IndexFlatIP (small dataset)")
        return faiss.IndexFlatIP(dim)

    # LARGE DATA → use IVF
    nlist = min(100, n_vectors // 2) 
    m = 8

    print(f"🚀 Using IVF index (nlist={nlist})")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)

    if embeddings is not None:
        index.train(embeddings)

    return index


def update_faiss(vectors):
    new_file_names, new_metadata, new_embeddings = vectors

    if len(new_embeddings) == 0:
        return

    new_embeddings = new_embeddings.astype("float32")
    dim = new_embeddings.shape[1]

    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        id_map = load_mapping()
        start_id = len(id_map)
    else:
        index = get_faiss_index(dim, new_embeddings)
        id_map = {}
        start_id = 0

    if hasattr(index, "is_trained") and not index.is_trained:
        if new_embeddings.shape[0] >= index.nlist:
            index.train(new_embeddings)
        else:
            print("⚠️ Not enough data to train IVF yet, skipping training")

    index.add(new_embeddings)

    for i, fname in enumerate(new_file_names):
        id_map[str(start_id + i)] = str(fname)

    faiss.write_index(index, FAISS_INDEX_FILE)
    save_mapping(id_map)

    print(f"✅ Added {len(new_embeddings)} vectors to FAISS")


def mark_processed(file_names):
    conn = sqlite3.connect('photo_metadata.db')
    cursor = conn.cursor()

    cursor.executemany("""
        UPDATE metadata 
        SET processed = 1 
        WHERE file_path = ?
    """, [(str(f),) for f in file_names])

    conn.commit()
    conn.close()
