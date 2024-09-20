from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from sqlalchemy.future import select
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from typing import List, Dict, AsyncGenerator
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from database import SessionLocal
from models import LecturerAnswer, StudentAnswer
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import re
import signal
import sys

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    await shutdown_event()

app = FastAPI(lifespan=lifespan)

# Tambahkan fungsi shutdown event
async def shutdown_event():
    print("Shutting down gracefully...")
    # Lakukan pembersihan di sini jika diperlukan
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    print("Shutdown complete.")

# Tambahkan event handler untuk startup dan shutdown
@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_event()))

@app.on_event("shutdown")
async def shutdown():
    await shutdown_event()

class TextProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

    def preprocess_text(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text)
        words = []
        for token in tokens:
            if token.startswith('##'):
                words[-1] += token[2:]
            else:
                words.append(token)
        return ' '.join(words)

class SubjectiveEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # NLI model for classification
        self.nli_model_name = "LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name).to(self.device)
        
        # SentenceTransformer model for sentence embedding
        self.sentence_model = SentenceTransformer("cassador/indobert-base-p2-nli-v1")
        self.negation_words = ["tidak", "bukan", "tanpa", "belum", "jangan"]
        self.exception_phrases = ["tidak hanya", "tidak lain", "tidak lain tidak bukan"]
        self.negation_pattern = re.compile(r'\b(?:' + '|'.join(self.negation_words) + r')\s+\w{4,}')

    def detect_negation(self, text: str) -> bool:
        text = text.lower()
        if any(phrase in text for phrase in self.exception_phrases):
            return False
        return bool(self.negation_pattern.search(text))

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def evaluate(self, lecturer_text: str, student_text: str) -> Dict[str, float]:
        # NLI evaluation
        inputs = self.nli_tokenizer(lecturer_text, student_text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            nli_outputs = self.nli_model(**inputs)

        nli_scores = torch.softmax(nli_outputs.logits, dim=1).squeeze().tolist()
        
        # Sentence embedding similarity
        embedding_lecturer = self.sentence_model.encode(lecturer_text)
        embedding_student = self.sentence_model.encode(student_text)
        semantic_similarity = self.cosine_similarity(embedding_lecturer, embedding_student) * 100

        # Combine NLI and semantic similarity
        nli_similarity = (1 - nli_scores[2]) * 100  # Inverse of contradiction score
        combined_similarity = (nli_similarity + semantic_similarity) / 2
        
        # Negation detection
        if self.detect_negation(lecturer_text) != self.detect_negation(student_text):
            combined_similarity *= 0.1
        
        return {
            'similarity': combined_similarity,
            'relevance': semantic_similarity,
            'coherence': min(combined_similarity, 100),
        }

class EssayComparer:
    def __init__(self, text_processor: TextProcessor, subjective_evaluator: SubjectiveEvaluator):
        self.text_processor = text_processor
        self.subjective_evaluator = subjective_evaluator

    def process_student(self, student_answer, lecturer_answer):
        cleaned_student_text = self.text_processor.preprocess_text(student_answer.cfile)
        cleaned_lecturer_text = self.text_processor.preprocess_text(lecturer_answer.answer_text)
        
        scores = self.subjective_evaluator.evaluate(cleaned_lecturer_text, cleaned_student_text)
        
        weights = {'similarity': 0.6, 'relevance': 0.2, 'coherence': 0.2}
        final_score = sum(weights[key] * scores[key] for key in weights if key in scores)
        
        return {
            'student_id': student_answer.npm,
            'student_answer': cleaned_student_text,
            'lecturer_answer': cleaned_lecturer_text,
            'score': round(final_score, 2),
            'detail': {key: round(value, 2) for key, value in scores.items()}
        }

    async def process_student_batch(self, student_answer_batch, lecturer_answer):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            tasks = [loop.run_in_executor(pool, self.process_student, answer, lecturer_answer) for answer in student_answer_batch]
            results = await asyncio.gather(*tasks)
        return results

    async def compare_essay(self, request: Request, db: AsyncSession):
        try:
            request_data = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")

        required_fields = ['cuserid', 'pertemuan', 'cacademic_year', 'pages']
        if not all(field in request_data for field in required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")

        cuser_id, pertemuan, cacademic_year, page = (request_data[field][0] for field in required_fields)
        page_size = 3

        try:
            # Perbaikan di sini
            result = await db.execute(
                select(LecturerAnswer).filter_by(cuserid=cuser_id, pertemuan=pertemuan, cacademic_year=cacademic_year)
            )
            lecturer_answer = result.scalar_one_or_none()

            if not lecturer_answer:
                raise HTTPException(status_code=404, detail="Lecturer answer not found")

            # Perbaikan di sini juga
            result = await db.execute(
                select(StudentAnswer).filter_by(cuserid=cuser_id, pertemuan=pertemuan, cacademic_year=cacademic_year)
            )
            student_answers = result.scalars().all()

            if not student_answers:
                raise HTTPException(status_code=404, detail="Student answers not found")

            start_index = (page - 1) * page_size
            paginated_student_answers = student_answers[start_index:start_index + page_size]

            results = await self.process_student_batch(paginated_student_answers, lecturer_answer)

            return {
                "message": "Successfully calculated scores",
                "data": results,
                "total_data": len(student_answers),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(student_answers) + page_size - 1) // page_size
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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

text_processor = TextProcessor()
subjective_evaluator = SubjectiveEvaluator()
essay_comparer = EssayComparer(text_processor, subjective_evaluator)
    
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session

@app.post("/compare-essay/")
async def compare_essay_endpoint(request: Request, db: AsyncSession = Depends(get_db)):
    return await essay_comparer.compare_essay(request, db)

class ModelTrainer:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "indobenchmark/indobert-base-p1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=4)

    async def get_training_data(self):
        student_result = await self.db.execute(select(StudentAnswer))
        lecturer_result = await self.db.execute(select(LecturerAnswer))
        
        student_answers = student_result.scalars().all()
        lecturer_answers = {answer.pertemuan: answer for answer in lecturer_result.scalars().all()}
        
        training_data = []
        for student_answer in student_answers:
            lecturer_answer = lecturer_answers.get(student_answer.pertemuan)
            if lecturer_answer:
                training_data.append({
                    'student_essay': student_answer.cfile,
                    'lecturer_essay': lecturer_answer.answer_text,
                    # Assume there's a score column, if not, you need to adjust this
                    'score': getattr(student_answer, 'score', 0)  
                })
        
        return training_data

    def tokenize_function(self, examples):
        return self.tokenizer(examples["combined_essay"], padding="max_length", truncation=True, max_length=512)

    async def fine_tune(self):
        training_data = await self.get_training_data()
        df = pd.DataFrame(training_data)
        
        if not df.empty:
            df['combined_essay'] = df.apply(lambda row: f"Student: {row['student_essay']} Lecturer: {row['lecturer_essay']}", axis=1)
        else:
            print("DataFrame is empty. Cannot add combined_essay column.")

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
        )
        
        trainer.train()
        
        self.model.save_pretrained("./fine_tuned_indobert_essay_scorer")
        self.tokenizer.save_pretrained("./fine_tuned_indobert_essay_scorer")

@app.post("/fine-tune-model/")
async def fine_tune_model(background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    try:
        background_tasks.add_task(run_fine_tuning)
        return {"message": "Fine-tuning started in the background"}
    except Exception as e:
        return {"error": f"Failed to start fine-tuning: {str(e)}"}

async def run_fine_tuning():
    try:
        print("Starting fine-tuning process...")
        # Implement fine-tuning process here
        print("Fine-tuning process completed.")
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
