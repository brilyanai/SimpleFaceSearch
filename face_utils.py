import numpy as np
from insightface.app import FaceAnalysis
import faiss
import cv2
import os

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])  # Bisa pilih model InsightFace
app.prepare(ctx_id=0)

def extract_embedding(img_path):
    img = cv2.imread(img_path)
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].embedding  # vektor 512 dimensi

def extract_multiple_embeddings(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    embeddings = []
    bboxes = []

    for face in faces:
        emb = face.embedding
        embeddings.append(emb)
        bboxes.append(face.bbox.astype(int))  # simpan posisi wajah

    return embeddings, bboxes

def extract_multiple_embeddings_with_crop(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)

    results = []
    for face in faces:
        emb = face.embedding
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        cropped_face = img[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.jpg', cropped_face)
        crop_bytes = buffer.tobytes()
        results.append({
            'embedding': emb,
            'bbox': bbox,
            'crop': crop_bytes  # simpan crop dalam bentuk bytes untuk tampilkan di template
        })

    return results


def save_faiss_index(index, path="database/faiss_index/index.faiss"):
    faiss.write_index(index, path)

def load_faiss_index(path="database/faiss_index/index.faiss"):
    if os.path.exists(path):
        return faiss.read_index(path)
    else:
        return faiss.IndexFlatL2(512)
