# API

Compare Essay Scorring API adalah sebuah layanan web berbasis FastAPI yang digunakan untuk membandingkan teks esai mahasiswa dengan teks dosen menggunakan metode Cosine Similarity berbasis model IndoBERT. Proyek ini memungkinkan penilaian otomatis esai berdasarkan kesamaan konten dengan jawaban dosen.

# Fitur

- **Cosine Similarity Calculation**: Menghitung kesamaan antara jawaban dosen dan mahasiswa menggunakan model IndoBERT.
- **Optimized Text Processing**: Menggunakan prapemrosesan teks yang efisien untuk mempercepat komputasi.
- **Asynchronous Processing**: Memproses beberapa esai mahasiswa secara asynchronous untuk meningkatkan performa.

## Teknologi yang Digunakan

- **FastAPI**: Web framework untuk membangun API yang cepat dan sederhana.
- **SQLAlchemy**: ORM yang digunakan untuk berinteraksi dengan basis data secara asinkron.
- **IndoBERT**: Model pre-trained dari Hugging Face digunakan untuk mendapatkan embedding dari teks.
- **NLTK**: Digunakan untuk prapemrosesan teks dan menghilangkan stop words dalam Bahasa Indonesia.
- **Torch**: Library untuk komputasi tensor dan menjalankan model machine learning.

## Instalasi

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1. **Clone Repository**
   
2. **Buat Virtual Environment**
   python -m venv venv
   source venv/bin/activate  # Untuk macOS/Linux
   .\venv\Scripts\activate  # Untuk Windows
   
3. **Instal Dependensi**
   pip install -r requirements.txt

4. **Setup Database**
   - Sesuaikan konfigurasi database di file `database.py` sesuai dengan setup Anda.
   - Jalankan migrasi atau buat database baru.

5. **Jalankan Aplikasi**
   uvicorn main:app --reload

6. **Akses API**
   - Aplikasi akan berjalan di `http://127.0.0.1:8000`.
   - Anda dapat mengakses dokumentasi API di `http://127.0.0.1:8000/docs`.

## Penggunaan

Untuk menggunakan API, Anda dapat mengirimkan POST request ke endpoint `/compare-essay/` dengan data dalam format JSON seperti berikut:

{
    "cuserid": ["TS13313aA"],
    "pertemuan": [1],
    "cacademic_year": ["2013/2014"]
}

Response akan mengembalikan hasil perhitungan nilai esai mahasiswa terhadap jawaban dosen.

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini:

1. Fork repository ini.
2. Buat branch fitur (`git checkout -b fitur/AmazingFeature`).
3. Commit perubahan Anda (`git commit -m 'Menambahkan fitur yang luar biasa'`).
4. Push ke branch (`git push origin fitur/AmazingFeature`).
5. Buat Pull Request.

## Lisensi

Proyek ini dilisensikan di bawah lisensi MIT - lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.
