# ================== INSTALL ==================
# !pip install -q transformers sentence-transformers faiss-cpu pymupdf gradio torch requests

import os, traceback, torch, fitz, re, random
import numpy as np
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from functools import lru_cache

# ================== CONFIG ==================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
FALLBACK_MODEL = "google/flan-t5-small"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 800, 200, 5

# ================== HELPERS ==================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join([p.get_text("text") for p in doc])
    except Exception as e:
        print("extract_text_from_pdf_bytes error:", e)
        return ""

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start, L = [], 0, len(text)
    while start < L:
        end = min(start + size, L)
        chunks.append(text[start:end])
        if end == L: break
        start = end - overlap
    return chunks

@lru_cache(maxsize=1)
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_texts(texts):
    model = load_embedder()
    embs = model.encode(texts, convert_to_numpy=True)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return embs.astype("float32")

def build_faiss(embs):
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

def try_load_pipe():
    try:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text2text-generation", model=FALLBACK_MODEL, device=device)
    except Exception as e:
        print("Generator load failed:", e)
        return None

GEN_PIPE = try_load_pipe()

# ================== STATE ==================
STATE = {"chunks": [], "index": None, "answers": [], "quiz": [], "topics": []}

# ================== FUNCTIONS ==================
def index_pdfs(files):
    texts = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                t = extract_text_from_pdf_bytes(fh.read())
                texts += chunk_text(t)
        except Exception as e:
            print("index_pdfs: error reading", f, e)
            continue
    if not texts:
        return "❌ No text extracted."
    embs = embed_texts(texts)
    idx = build_faiss(embs)
    STATE.update({"chunks": texts, "index": idx})
    return f"✅ Indexed {len(texts)} chunks from {len(files)} file(s)."

def build_prompt(q, chunks, marks=None):
    context = "\n\n".join(chunks)

    if marks == 2:
        style = "Write only 3–4 concise bullet points."
        max_len = 80
    elif marks == 3:
        style = "Write exactly 5–6 bullet points."
        max_len = 120
    elif marks == 8:
        style = "Write 10–12 detailed bullet points."
        max_len = 250
    elif marks == 16:
        return context, 0  

    return f"""You are StudyMate, an expert teacher.
Strictly follow the marks scheme.

Question: {q}
Marks: {marks}
Instruction: {style}

Context:
{context}

Answer as bullet points:""", max_len

def generate_answer(prompt, max_len):
    if max_len == 0:   # 16 marks → return full context
        return prompt
    if GEN_PIPE:
        try:
            out = GEN_PIPE(prompt, max_new_tokens=max_len)
            if isinstance(out, list):
                text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
                return text.strip()
        except Exception as e:
            print("Pipeline error:", e)
    return "⚠ No generator available."

def ask_question(q, marks):
    if STATE["index"] is None:
        return "❌ No PDFs indexed.", ""
    query_emb = embed_texts([q])
    D, I = STATE["index"].search(query_emb, TOP_K)
    idxs = [int(i) for i in I[0] if i != -1]
    retrieved = [STATE["chunks"][i] for i in idxs] if idxs else []
    if not retrieved:
        return "No relevant context found.", ""
    prompt, max_len = build_prompt(q, retrieved, marks)
    ans = generate_answer(prompt, max_len)
    return ans, ans

def save_answer(q, ans, marks):
    if not ans.strip():
        return "⚠ No answer to save."
    STATE["answers"].append({"q": q, "a": ans, "marks": marks})
    STATE["topics"].append(q)
    return "✅ Answer saved!"

# --------- QUIZ (MCQs) -----------
def generate_mcq_from_context(context, n=3):
    """Generate n MCQs from context text."""
    if not GEN_PIPE:
        return ["⚠ Generator not available."]
    prompt = f"""
You are StudyMate. Create {n} multiple-choice questions (MCQs) from the following context.
Each question must have:
- A clear question
- Four complete options: A, B, C, D
- The correct answer clearly labeled as "Answer: <option letter>"
Ensure all words and sentences are fully complete.

Context:
{context}

MCQs:
"""
    try:
        out = GEN_PIPE(prompt, max_new_tokens=400)
        if isinstance(out, list):
            text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
            return text.strip().split("\n\n")
    except Exception as e:
        print("MCQ generation error:", e)
    return ["⚠ Could not generate MCQs."]

def conduct_quiz():
    if not STATE["chunks"]:
        return "❌ Index PDFs first!"
    # Use first few chunks for quiz
    selected = STATE["chunks"][:2]
    quiz_qs = []
    for chunk in selected:
        mcqs = generate_mcq_from_context(chunk, n=2)
        quiz_qs.extend(mcqs)
    STATE["quiz"] = quiz_qs
    return "\n\n".join(quiz_qs)

def tracker():
    return {
        "Topics Discussed": STATE["topics"],
        "Quiz Questions": STATE["quiz"],
        "Answers Saved": len(STATE["answers"])
    }

def get_saved_answers():
    if not STATE["answers"]:
        return [["-", "-", "-", "-"]]
    rows = []
    for i, ans in enumerate(STATE["answers"], 1):
        rows.append([i, ans["q"], ans["a"], ans["marks"]])
    return rows

# ================== GRADIO UI ==================
with gr.Blocks() as demo:
    gr.Markdown("# 📘 StudyMate — Marks-based Bullet Point Answers + MCQ Quiz")

    with gr.Tab("Index PDFs"):  
        pdfs = gr.File(file_types=[".pdf"], file_count="multiple", type="filepath")  
        idx_btn = gr.Button("Index")  
        idx_out = gr.Textbox()  
        idx_btn.click(index_pdfs, pdfs, idx_out)  

    with gr.Tab("Q&A"):  
        q = gr.Textbox(label="Ask Question")  
        marks = gr.Radio([2,3,8,16], label="Marks", value=2)  
        go = gr.Button("Get Answer")  
        ans = gr.Markdown(label="Answer")  
        ctx = gr.Textbox(label="Context (Answer will appear here)", lines=12)  

        save_btn = gr.Button("💾 Save This Answer")  
        save_status = gr.Textbox(label="Save Status")  

        go.click(ask_question, [q, marks], [ans, ctx])  
        save_btn.click(save_answer, [q, ans, marks], save_status)

    with gr.Tab("Quiz"):  
        quiz_btn = gr.Button("Generate MCQ Quiz")  
        quiz_out = gr.Textbox(lines=15)  
        quiz_btn.click(conduct_quiz, outputs=quiz_out)  

    with gr.Tab("Saved Answers"):  
        refresh_btn = gr.Button("Refresh Saved Answers")  
        table = gr.Dataframe(headers=["ID", "Question", "Answer", "Marks"], wrap=True)  
        refresh_btn.click(get_saved_answers, outputs=table)  

    with gr.Tab("Tracker"):  
        track_btn = gr.Button("Show Tracker")  
        track_out = gr.JSON()  
        track_btn.click(tracker, outputs=track_out)  

print("Launching Gradio... (share=True gives public link)")
demo.launch(share=True)