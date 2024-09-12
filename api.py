from fastapi import FastAPI, Request, HTTPException, Depends
from sqlalchemy.future import select
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import re
from typing import List, Dict, Tuple, AsyncGenerator
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from database import SessionLocal
from models import AnswerDosen, AnswerMahasiswa
from fastapi.middleware.cors import CORSMiddleware
import nltk
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

origins = [
    "http://localhost",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained DistilBERT model and tokenizer once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1").to(device)

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('indonesian'))

def get_text_embedding(texts: Tuple[str]) -> np.ndarray:
    inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def preprocess_text(text: str) -> str:
    return re.sub(r'\d+\.\s*', '', text.lower())

def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def calculate_cosine_similarity_bert(embedding1: np.ndarray, embedding2: np.ndarray) -> Dict[str, float]:
    dot_product = np.dot(embedding1, embedding2.T)  # Transpose embedding2 to align dimensions
    norm_vec1 = np.linalg.norm(embedding1)
    norm_vec2 = np.linalg.norm(embedding2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = dot_product / (norm_vec1 * norm_vec2)

    return {
        'dotProduct': float(dot_product),
        'length1': float(norm_vec1),
        'length2': float(norm_vec2),
        'cosineSimilarity': float(cosine_sim)
    }

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session

def process_mahasiswa(jawaban_mahasiswa, embeddings_dosen):
    cleaned_text = preprocess_text(jawaban_mahasiswa.cfile)
    chunked_texts = chunk_text(cleaned_text)
    embeddings = np.mean([get_text_embedding(tuple([chunk])) for chunk in chunked_texts], axis=0)
    nilai = calculate_cosine_similarity_bert(embeddings_dosen, embeddings)
    return {
        'npm': jawaban_mahasiswa.npm,
        'jawaban_mahasiswa': cleaned_text,
        'score': nilai['cosineSimilarity']
    }

async def process_mahasiswa_batch(jawaban_mahasiswa_batch, embeddings_dosen):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        tasks = [loop.run_in_executor(pool, process_mahasiswa, jawaban, embeddings_dosen) for jawaban in jawaban_mahasiswa_batch]
        results = await asyncio.gather(*tasks)
    return results

@app.post("/compare-essay/")
async def compare_essay(request: Request, db: AsyncSession = Depends(get_db)):
    request_data = await request.json()

    if 'cuserid' in request_data and 'pertemuan' in request_data and 'cacademic_year' in request_data and 'pages' in request_data:
        cuser_id = request_data['cuserid'][0]
        pertemuan = request_data['pertemuan'][0]
        cacademic_year = request_data['cacademic_year'][0]
        page = request_data['pages'][0]
        page_size = 3  # Set the number of items per page

        try:
            # Fetching data from the database
            jawaban_dosen_result = await db.execute(
                select(AnswerDosen).filter_by(cuserid=cuser_id, pertemuan=pertemuan, cacademic_year=cacademic_year)
            )
            jawaban_dosen = jawaban_dosen_result.scalar_one_or_none()

            if not jawaban_dosen:
                raise HTTPException(detail="Jawaban dosen tidak ditemukan")

            jawaban_mahasiswa_result = await db.execute(
                select(AnswerMahasiswa).filter_by(cuserid=cuser_id, pertemuan=pertemuan, cacademic_year=cacademic_year)
            )
            jawaban_mahasiswa = jawaban_mahasiswa_result.scalars().all()

            if not jawaban_mahasiswa:
                raise HTTPException(status_code=404, detail="Jawaban mahasiswa tidak ditemukan")

            # Preprocess texts
            cleaned_text_dosen = preprocess_text(jawaban_dosen.answer_text)
            chunked_texts_dosen = chunk_text(cleaned_text_dosen)
            embeddings_dosen = np.mean([get_text_embedding(tuple([chunk])) for chunk in chunked_texts_dosen], axis=0)

            # Pagination logic
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            paginated_jawaban_mahasiswa = jawaban_mahasiswa[start_index:end_index]

            # Process mahasiswa answers in batches
            results = await process_mahasiswa_batch(paginated_jawaban_mahasiswa, embeddings_dosen)

            return {
                "message": "Berhasil menghitung nilai",
                "data": results,
                "total_data": len(jawaban_mahasiswa),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(jawaban_mahasiswa) + page_size - 1) // page_size
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Gagal menghitung nilai")