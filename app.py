# ================== INSTALL ==================
!pip install -q transformers sentence-transformers faiss-cpu pymupdf gradio torch requests graphviz

import os, traceback, requests, torch, fitz, json
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
        pipe = pipeline("text2text-generation", model=FALLBACK_MODEL, device=device)
        print("Loaded generator:", FALLBACK_MODEL)
        return pipe
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
    style = ""
    if marks == 2:
        style = "Give only formula + 2 short sentences."
    elif marks == 3:
        style = "Give formula + 5-6 key points."
    elif marks == 8:
        style = "Give a brief structured answer with example."
    elif marks == 16:
        style = "Give detailed answer with example and explanation."
    return f"""You are StudyMate, an expert teacher.
Use the context strictly to write a complete, clear answer.

Question: {q}
Marks: {marks}
Style: {style}

Context:
{context}

Answer in full sentences:"""

def generate_answer(prompt):
    if GEN_PIPE:
        try:
            out = GEN_PIPE(prompt, max_new_tokens=400)
            if isinstance(out, list):
                return out[0].get("generated_text") or out[0].get("text") or str(out[0])
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
    prompt = build_prompt(q, retrieved, marks)
    ans = generate_answer(prompt)
    return ans, "\n\n".join(retrieved)

def save_answer(q, ans, marks):
    if not ans.strip():
        return "⚠ No answer to save."
    STATE["answers"].append({"q": q, "a": ans, "marks": marks})
    STATE["topics"].append(q)
    return "✅ Answer saved!"

def conduct_quiz():
    if not STATE["chunks"]:
        return "Index PDFs first!"
    qs = ["Q"+str(i+1)+": "+STATE["chunks"][i][:80] for i in range(min(5,len(STATE["chunks"])))]
    STATE["quiz"] = qs
    return "\n".join(qs)

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
    gr.Markdown("# 📘 StudyMate — Full Answer Version")

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
        ctx = gr.Textbox(label="Context", lines=6)  

        save_btn = gr.Button("💾 Save This Answer")  
        save_status = gr.Textbox(label="Save Status")  

        go.click(ask_question, [q, marks], [ans, ctx])  
        save_btn.click(save_answer, [q, ans, marks], save_status)

    with gr.Tab("Quiz"):  
        quiz_btn = gr.Button("Conduct Quiz (5 Qs)")  
        quiz_out = gr.Textbox(lines=8)  
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
