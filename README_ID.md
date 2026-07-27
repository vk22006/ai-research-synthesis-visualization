# Sintesis Riset AI & Pembuat Graf Pengetahuan (AI Research Synthesis & Knowledge Graph Builder)

[English](README.md) | [தமிழ்](README_TA.md) | [中文](README_ZH.md) | [हिन्दी](README_HI.md) | Bahasa Indonesia

![Visualisasi utama aplikasi](assets/app_visualization.gif)

Sebuah alat end-to-end untuk mempercepat riset AI dengan mengambil makalah secara otomatis, menyintesis kontennya, dan mengorganisasikannya ke dalam graf pengetahuan (knowledge graph) interaktif.

Aplikasi ini menggunakan model Pemrosesan Bahasa Alami (NLP) untuk mengekstrak ringkasan dan klaim utama dari makalah arXiv serta memvisualisasikan hubungan antar makalah riset berdasarkan kemiripan semantik.

## Fitur Utama

- **Automated Research Retrieval**: Mengambil makalah riset terbaru langsung dari arXiv berdasarkan topik pencarian Anda.
- **AI-Powered Synthesis**: Meringkas abstrak makalah secara otomatis dan mengekstrak klaim/kontribusi utama menggunakan Hugging Face Transformers.
- **Semantic Similarity Analysis**: Menghitung kemiripan semantik antar makalah menggunakan Sentence-Transformers untuk menemukan keterkaitan.
- **Interactive Knowledge Graphs**: Membangun dan menampilkan graf pengetahuan interaktif menggunakan NetworkX dan Pyvis, yang mengilustrasikan bagaimana berbagai makalah riset saling berhubungan.
- **Modern Dashboard**: Antarmuka Streamlit yang intuitif untuk interaksi yang mulus, konfigurasi pencarian, dan eksplorasi visualisasi.
- **Robust Backend API**: Arsitektur backend berbasis FastAPI yang menangani alur kerja mulai dari pengambilan data hingga pembuatan graf.

Untuk pembaruan terkini, lihat [Catatan Pembaruan (UPDATE_LOG)](UPDATE_LOG.md).

## Struktur Proyek

```
.
├── app.py                      # Aplikasi backend FastAPI
├── requirements.txt            # Dependensi Python
├── backend/                    # Logika utama backend
│   ├── fetch_papers.py         # Pengambilan data arXiv
│   ├── summarize.py            # Peringkasan abstrak
│   ├── claim_extractor.py      # Pengambilan klaim utama
│   ├── embeddings.py           # Perhitungan matriks kemiripan
│   ├── graph_builder.py        # Pembuatan graf pengetahuan
│   └── graph_visualizer.py     # Visualisasi HTML graf
├── frontend/                   # Antarmuka Pengguna (Frontend UI)
│   └── streamlit_app.py        # Aplikasi dashboard Streamlit
├── lib/                        # Utilitas/modul tambahan
└── data/                       # Direktori untuk hasil yang dihasilkan (misalnya, graph.html)
```

## Teknologi yang Digunakan

- **Kerangka Kerja Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Antarmuka Frontend**: [Streamlit](https://streamlit.io/)
- **NLP & Embeddings**: [Transformers](https://huggingface.co/docs/transformers/index), [Sentence-Transformers](https://sbert.net/), [PyTorch](https://pytorch.org/)
- **Graf & Visualisasi**: [NetworkX](https://networkx.org/), [Pyvis](https://pyvis.readthedocs.io/)
- **Pemrosesan Data**: [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)

## Memulai

### Prasyarat

Pastikan Anda telah menginstal Python 3.8+. Disarankan untuk menggunakan lingkungan virtual (virtual environment).

### Instalasi

1. Klon repositori ini atau buka direktori proyek.
2. Instal dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

### Menjalankan Aplikasi

Aplikasi ini terdiri dari API backend dan dashboard frontend. Anda perlu menjalankan keduanya secara bersamaan.

#### 1. Jalankan Backend (FastAPI)

Jalankan server FastAPI menggunakan `uvicorn` (dari direktori utama):

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API backend akan tersedia di `http://localhost:8000`. Anda dapat melihat dokumentasi API di `http://localhost:8000/docs`.

#### 2. Jalankan Frontend (Streamlit)

Buka jendela terminal baru, lalu jalankan aplikasi Streamlit:

```bash
streamlit run frontend/streamlit_app.py
```

Dashboard frontend akan terbuka secara otomatis di peramban (browser) default Anda di `http://localhost:8501`.

## Cara Penggunaan

1. Buka frontend Streamlit.
2. Pada bilah samping (sidebar), masukkan **Topik Riset (Research Topic)** (misalnya, "Large Language Models", "Quantum Machine Learning", "Retrieval-Augmented Generation").
3. Sesuaikan **Jumlah Hasil Maksimal (Max Results)** (berapa banyak makalah yang diambil) dan **Ambang Batas Kemiripan (Similarity Threshold)** (skor kemiripan minimum untuk membentuk hubungan pada graf).
4. Klik **Jalankan Analisis (Run Analysis)**.
5. Sistem akan memproses makalah dan menampilkan Makalah Terkstrak, Ringkasan, Klaim yang diekstrak, serta visualisasi Graf Pengetahuan Interaktif.

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT, lihat [LICENSE](LICENSE) untuk rincian lebih lanjut.
