from fastapi import FastAPI, Request, HTTPException, Depends
from sqlalchemy.future import select
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import re
from typing import List, Dict, Tuple, AsyncGenerator
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from database import SessionLocal, Base  # Import from your database setup file
from models import AnswerDosen, AnswerMahasiswa  # Import your models
from fastapi.middleware.cors import CORSMiddleware
from spellchecker import SpellChecker
import nltk
from nltk.corpus import stopwords

app = FastAPI()

origins = [
    "http://localhost",
    "http://127.0.0.1",
    # Tambahkan domain lain jika diperlukan
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load pre-trained IndoBERT model and tokenizer once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-large-p2")
model = AutoModel.from_pretrained("indobenchmark/indobert-large-p2").to(device)

nltk.download('stopwords')
nltk.download('punkt')

STOP_WORDS = set(stopwords.words('indonesian'))
spell = SpellChecker()

def get_text_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def preprocess_text(text: str) -> str:
    text = re.sub(r'\d+', '', text)
    return text

def dot_product_calculate(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2)

def norm_calculate(vec: np.ndarray) -> float:
    return np.linalg.norm(vec)

def cosine_similarity(tokens1: List[str], tokens2: List[str]) -> Tuple[float, float, float, float]:
    unique_tokens = list(set(tokens1) | set(tokens2))
    freq_vec1 = np.array([tokens1.count(token) for token in unique_tokens])
    freq_vec2 = np.array([tokens2.count(token) for token in unique_tokens])

    dot_product = dot_product_calculate(freq_vec1, freq_vec2)
    norm_vec1 = norm_calculate(freq_vec1)
    norm_vec2 = norm_calculate(freq_vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError("One of the vectors has norm 0, cosine similarity is undefined.")
    
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return dot_product, cosine_sim, norm_vec1, norm_vec2

def cosine_similarity_bert(vec1: np.ndarray, vec2: np.ndarray) -> Tuple[float, float, float, float]:
    dot_product = dot_product_calculate(vec1, vec2)
    norm_vec1 = norm_calculate(vec1)
    norm_vec2 = norm_calculate(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError("One of the vectors has norm 0, cosine similarity is undefined.")
    
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return dot_product, cosine_sim, norm_vec1, norm_vec2

def calculate_cosine_similarity_bert(text1: str, text2: str) -> Dict[str, float]:
    embedding1 = get_text_embedding(text1)
    embedding2 = get_text_embedding(text2)
    
    dot_product, cosine_sim, norm_vec1, norm_vec2 = cosine_similarity_bert(embedding1, embedding2)
    
    return {
        'dotProduct': float(dot_product),
        'length1': float(norm_vec1),
        'length2': float(norm_vec2),
        'cosineSimilarity': float(cosine_sim)
    }


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session

@app.post("/compare-essay/")
async def compare_essay(request: Request, db: AsyncSession = Depends(get_db)):
    request_data = await request.json()

    if 'cuserid' in request_data and 'pertemuan' in request_data and 'cacademic_year' in request_data:
        cuser_id = request_data['cuserid'][0]
        pertemuan = request_data['pertemuan'][0]
        cacademic_year = request_data['cacademic_year'][0]

        try:
            jawaban_dosen_result = await db.execute(
                select(AnswerDosen).filter_by(cuserid=cuser_id, pertemuan=pertemuan, cacademic_year=cacademic_year)
            )
            jawaban_dosen = jawaban_dosen_result.scalar_one_or_none()

            if not jawaban_dosen:
                raise HTTPException(detail="Jawaban dosen tidak ditemukan")

            jawaban_mahasiswa_result = await db.execute(
                select(AnswerMahasiswa)
                .filter_by(cuserid=cuser_id, pertemuan=pertemuan, cacademic_year=cacademic_year)
            )
            jawaban_mahasiswa = jawaban_mahasiswa_result.scalars().all()

            if not jawaban_mahasiswa:
                raise HTTPException(status_code=404, detail="Jawaban mahasiswa tidak ditemukan")

            tokens_dosen = re.findall(r'\w+', jawaban_dosen.answer_text.lower())
            keywords = list(set(tokens_dosen) - STOP_WORDS)

            async def process_student_answer(jawaban):
                npm = jawaban.npm
                regex_text = re.sub(r'\b\d+\.\s*', '', jawaban.cfile)
                regex_text_dosen = re.sub(r'\b\d+\.\s*', '', jawaban_dosen.answer_text)

               
             
                nilai = calculate_cosine_similarity_bert(jawaban_dosen.answer_text, regex_text)

               

                return {
                    'kd_matkul': cuser_id,
                    'npm': npm,
                    'jawaban_dosen': regex_text_dosen,
                    'jawaban_mahasiswa': regex_text,
                    'score': nilai
                }

            tasks = [process_student_answer(jawaban) for jawaban in jawaban_mahasiswa]
            all_results = await asyncio.gather(*tasks)

            return {
                "message": "Berhasil menghitung nilai",
                "data": all_results,
                "total_data": len(jawaban_mahasiswa)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Gagal menghitung nilai")






