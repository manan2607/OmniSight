import numpy as np
import torch
import open_clip
from PIL import Image


EMBEDDING_FILE = "image_embeddings.npz"
TOP_K = 1

device = "mps" if torch.backends.mps.is_available() else "cpu"



model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()



data = np.load(EMBEDDING_FILE, allow_pickle=True)

embeddings = data["embeddings"]         
file_names = data["file_names"]
metadata = data["metadata"]

embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def text_to_embedding(query):
    tokens = tokenizer([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)

    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()[0]


def image_to_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy()[0]


def search(query_embedding, top_k=TOP_K):
    scores = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "file_name": file_names[idx],
            "score": float(scores[idx]),
            "metadata": metadata[idx]
        })

    return results

def search_by_text(query):
    query_embedding = text_to_embedding(query)
    return search(query_embedding)

def search_by_image(image_path):
    query_embedding = image_to_embedding(image_path)
    return search(query_embedding)


if __name__ == "__main__":
    
    query = "girl in a red dress with a DSLR camera"
    results = search_by_text(query)

    print("\n🔍 Text Search Results:")
    for r in results:
        print(r)