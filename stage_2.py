import sqlite3
import torch
import os
import open_clip
import numpy as np
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()


device = "mps" if torch.backends.mps.is_available() else "cpu"


model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()


def metadata_to_text(meta):
    return f"{Path(meta.get('file_name','')).name} {meta.get('date_time','')} {meta.get('address','')} {meta.get('lat','')} {meta.get('lon','')}"
     



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
        id, file_name, date_time, location,lat, lon, _ = row
        data.append({
            "id": id,
            "metadata": {
                "file_name": file_name,
                "date_time": date_time,
                "lat": lat,
                "lon": lon,
                "address": location
            }
        })
    
    return data




def process_batch(batch):
    images = []
    texts = []
    file_names = []
    metadata_list = []

    conn = sqlite3.connect('photo_metadata.db')
    cursor = conn.cursor()

    for item in batch:

        file_name = Path(item['metadata']['file_name'])
        img_path = f"photos/{file_name}"

        try:
            image = Image.open(img_path)

            if not isinstance(image, Image.Image):
                print(f"Invalid image type: {file_name}")
                continue

            image = image.convert("RGB")

            cursor.execute("""
                UPDATE metadata 
                SET processed = 1 
                WHERE file_path = ?
            """, (str(file_name),))
            conn.commit()

        except Exception as e:
            print(f"Error loading: {file_name}, {e}")
            continue

        if isinstance(image, list):
            print(f"Skipping list-type image: {file_name}")
            continue
        images.append(preprocess(image).unsqueeze(0))
        texts.append(metadata_to_text(item["metadata"]))
        file_names.append(file_name)
        metadata_list.append(item["metadata"])

    conn.close()

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
    combined = 0.65 * image_features + 0.35 * text_features

    return file_names, metadata_list, combined.cpu().numpy()


output_file = "image_embeddings.npz"


def npz_file_making(vectors):

    new_file_names, new_metadata, new_embeddings = vectors
    
    if os.path.exists(output_file):
        data = np.load(output_file, allow_pickle=True)
        old_embeddings = data["embeddings"]
        old_file_names = data["file_names"]
        old_metadata = data["metadata"]

        embeddings = np.vstack([old_embeddings, new_embeddings])
        file_names = np.concatenate([old_file_names, new_file_names])
        metadata = np.concatenate([old_metadata, new_metadata])

    else:
        embeddings = new_embeddings
        file_names = new_file_names
        metadata = new_metadata

    np.savez(
        output_file,
        embeddings=embeddings,
        file_names=file_names,
        metadata=metadata
    )

    return
