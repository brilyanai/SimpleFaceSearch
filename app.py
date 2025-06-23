from flask import Flask, request, render_template, redirect, url_for
import os
import uuid
import numpy as np
from database.connection import get_connection
from base64 import b64encode
from face_utils import extract_embedding, load_faiss_index, save_faiss_index, extract_multiple_embeddings, extract_multiple_embeddings_with_crop
from flask import send_from_directory
from PIL import Image
import io

# Konfigurasi Flask
app = Flask(__name__, template_folder='frontend')
UPLOAD_FOLDER = 'data_faces'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder upload dan index faiss ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('database/faiss_index', exist_ok=True)

# Load FAISS index
faiss_index = load_faiss_index()

# ===================== ROUTES =====================

# [1] Halaman Utama - List semua wajah
@app.route('/')
def index():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT f.id, f.name, COUNT(e.id) as photo_count
        FROM faces f
        LEFT JOIN face_embeddings e ON f.id = e.face_id
        GROUP BY f.id, f.name
    """)
    faces = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('index.html', faces=faces)

# [2] Upload Wajah Baru
# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         name = request.form['name']
#         file = request.files['image']
#         if file:
#             # Simpan gambar
#             filename = str(uuid.uuid4()) + "_" + file.filename
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             # Ekstraksi embedding wajah
#             emb = extract_embedding(file_path)
#             if emb is None:
#                 return "Wajah tidak terdeteksi"

#             # Simpan ke FAISS
#             faiss_index.add(np.array([emb], dtype='float32'))
#             save_faiss_index(faiss_index)

#             # Simpan metadata ke database
#             conn = get_connection()
#             cursor = conn.cursor()
#             cursor.execute("INSERT INTO faces (name, image_path) VALUES (%s, %s)", (name, file_path))
#             conn.commit()
#             cursor.close()
#             conn.close()

#             return redirect(url_for('index'))

#     return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        name = request.form.get('name')
        file = request.files.get('image')

        if not name or not file:
            return "Nama dan file gambar wajib diisi.", 400

        # Simpan gambar asli
        filename = str(uuid.uuid4()) + "_" + file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Ambil semua wajah yang terdeteksi
        results = extract_multiple_embeddings_with_crop(image_path)
        if not results:
            return "Tidak ada wajah terdeteksi di gambar yang diupload.", 400

        # Simpan ke DB
        conn = get_connection()
        cursor = conn.cursor()

        # Cek apakah nama sudah ada
        cursor.execute("SELECT id FROM faces WHERE name = %s", (name,))
        row = cursor.fetchone()
        if row:
            face_id = row[0]
        else:
            cursor.execute("INSERT INTO faces (name) VALUES (%s)", (name,))
            face_id = cursor.lastrowid

        # Simpan semua embedding + crop
        for i, result in enumerate(results):
            emb = result['embedding']
            crop_path = None

            if 'crop' in result and result['crop'] is not None:
                crop_filename = f"{uuid.uuid4()}_crop_{i}.jpg"
                crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
                crop_data = result['crop']

                # Cek apakah crop berupa bytes atau image
                if isinstance(crop_data, bytes):
                    image = Image.open(io.BytesIO(crop_data))
                    image.save(crop_path)
                elif hasattr(crop_data, 'save'):
                    crop_data.save(crop_path)

            # Simpan ke face_embeddings
            cursor.execute(
                "INSERT INTO face_embeddings (face_id, embedding, image_path) VALUES (%s, %s, %s)",
                (face_id, emb.astype(np.float32).tobytes(), crop_path)
            )

        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for('index'))

    return render_template('upload.html')

# [3] Edit Nama Wajah
@app.route('/faces/<int:face_id>/edit', methods=['GET', 'POST'])
def edit_face(face_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        new_name = request.form['name']
        cursor.execute("UPDATE faces SET name = %s WHERE id = %s", (new_name, face_id))
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for('face_detail', face_id=face_id))

    cursor.execute("SELECT * FROM faces WHERE id = %s", (face_id,))
    face = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template('edit.html', face=face)

# [4] Hapus Wajah
@app.route('/faces/<int:face_id>/delete', methods=['POST'])
def delete_face(face_id):
    conn = get_connection()
    cursor = conn.cursor()

    # Hapus foto dari file
    cursor.execute("SELECT image_path FROM face_embeddings WHERE face_id = %s", (face_id,))
    paths = cursor.fetchall()
    for p in paths:
        if p[0] and os.path.exists(p[0]):
            os.remove(p[0])

    # Hapus dari DB
    cursor.execute("DELETE FROM face_embeddings WHERE face_id = %s", (face_id,))
    cursor.execute("DELETE FROM faces WHERE id = %s", (face_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for('index'))

@app.route('/faces/<int:face_id>')
def face_detail(face_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Ambil nama orang
    cursor.execute("SELECT * FROM faces WHERE id = %s", (face_id,))
    face = cursor.fetchone()

    # Ambil semua foto yang terkait
    cursor.execute("SELECT * FROM face_embeddings WHERE face_id = %s", (face_id,))
    photos = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('detail.html', face=face, photos=photos)

@app.route('/data_faces/<path:filename>')
def data_faces(filename):
    return send_from_directory('data_faces', filename)

# @app.route('/search', methods=['GET', 'POST'])
# def search():
#     if request.method == 'POST':
#         file = request.files['image']
#         if file:
#             filename = str(uuid.uuid4()) + "_" + file.filename
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             emb = extract_embedding(file_path)
#             if emb is None:
#                 return "Wajah tidak terdeteksi di gambar."

#             # Cari embedding paling mirip (top 1)
#             D, I = faiss_index.search(np.array([emb], dtype='float32'), k=1)
#             if I[0][0] == -1:
#                 return "Tidak ada wajah cocok dalam database."

#             matched_id = I[0][0]

#             # Ambil metadata dari database berdasarkan ID index
#             conn = get_connection()
#             cursor = conn.cursor(dictionary=True)
#             cursor.execute("SELECT * FROM faces LIMIT 1 OFFSET %s", (int(matched_id),))
#             matched_face = cursor.fetchone()
#             cursor.close()
#             conn.close()

#             return render_template('search_result.html', face=matched_face, similarity=D[0][0], query_image=file_path)

#     return render_template('search.html')

THRESHOLD = 0.4  # cosine distance threshold

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return "Gambar wajib diunggah", 400

        # Simpan gambar sementara
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_search.jpg')
        file.save(image_path)

        # Deteksi semua wajah di gambar
        detected_faces = extract_multiple_embeddings_with_crop(image_path)
        if not detected_faces:
            return "Tidak ada wajah terdeteksi", 400

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        # Ambil semua embedding dari DB
        cursor.execute("""
            SELECT fe.*, f.name
            FROM face_embeddings fe
            JOIN faces f ON f.id = fe.face_id
        """)
        db_faces = cursor.fetchall()

        hasil = []
        for face in detected_faces:
            query_emb = face['embedding']
            crop_img = face.get('crop')

            # Bandingkan ke semua embedding di database
            best_match = None
            best_score = -1

            for db in db_faces:
                db_emb = np.frombuffer(db['embedding'], dtype=np.float32)
                sim = cosine_similarity(query_emb, db_emb)
                print(f"Data Similarity is {sim} - Best Score is {best_score}")
                if sim > best_score:
                    best_score = sim
                    best_match = db

            status = "MATCH" if best_score >= THRESHOLD else "TIDAK COCOK"

            crop_path = None

            if crop_img:
                crop_filename = f"crop_search_{np.random.randint(10000)}.jpg"
                crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)

                # Cek apakah crop berupa bytes atau image
                if isinstance(crop_img, bytes):
                    image = Image.open(io.BytesIO(crop_img))
                    image.save(crop_path)
                elif hasattr(crop_img, 'save'):
                    crop_img.save(crop_path)

                # crop_img.save(crop_path)

            # Tambahkan hasil ke tabel
            data_post = {
                'detected_crop': crop_path,
                'matched_name': best_match['name'] if best_match.get('name') and status == "MATCH" else "-",
                'matched_img': best_match['image_path'] if best_match.get('image_path') and status == "MATCH" else None,
                'score': round(best_score, 4) if best_score is not None else 0.0,
                'status': status
            }
            
            hasil.append(data_post)

        cursor.close()
        conn.close()
        print(f"Data Post is {hasil}")
        return render_template('search_result_multi.html', query_image=image_path, hasil=hasil)

    return render_template('search.html')

# ===================== MAIN =====================
if __name__ == '__main__':
    app.run(debug=True)
