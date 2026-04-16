import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import open_clip
import faiss
import json

from PIL import Image

FAISS_INDEX_FILE = "faiss.index"
FAISS_MAP_FILE = "faiss_map.json"
TOP_K = 1

device = "mps" if torch.backends.mps.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="laion2b_s32b_b82k"
)

tokenizer = open_clip.get_tokenizer("ViT-L-14")

model = model.to(device)
model.eval()

index = faiss.read_index(FAISS_INDEX_FILE)

with open(FAISS_MAP_FILE, "r") as f:
    id_map = json.load(f)

def text_to_embedding(query):
    tokens = tokenizer([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)

    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().astype("float32")

def image_to_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().astype("float32")


def search(query_embedding, top_k=TOP_K):
    # FAISS expects shape (1, dim)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "file_name": id_map[str(idx)],
            "score": float(score)
        })

    return results

def search_by_text(query):
    query_embedding = text_to_embedding(query)
    return search(query_embedding)

def search_by_image(image_path):
    query_embedding = image_to_embedding(image_path)
    return search(query_embedding)

if __name__ == "__main__":
    
    query = "image that was clicked in Goa in kitchen"
    results = search_by_text(query)

    print("\n🔍 Text Search Results:")
    for r in results:
        print(r)