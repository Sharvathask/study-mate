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
# Using IBM Granite model for text generation
GRANITE_MODEL = "ibm-granite/granite-3b-code-instruct"  # IBM Granite model
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

def try_load_granite():
    try:
        device = 0 if torch.cuda.is_available() else -1
        # Try loading IBM Granite model first
        pipe = pipeline("text-generation", model=GRANITE_MODEL, device=device, 
                       trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        print("Loaded IBM Granite model:", GRANITE_MODEL)
        return pipe
    except Exception as e:
        print("Granite model load failed, trying fallback:", e)
        try:
            # Fallback to FLAN-T5 if Granite fails
            fallback_pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=device)
            print("Loaded fallback model: google/flan-t5-small")
            return fallback_pipe
        except Exception as e2:
            print("All models failed to load:", e2)
            return None

GEN_PIPE = try_load_granite()

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
    
    # Enhanced prompt for Granite model
    return f"""You are StudyMate, an expert AI teacher powered by IBM Granite.
Use the provided context strictly to write a complete, clear, and accurate answer.

Question: {q}
Required Marks: {marks}
Answer Style: {style}

Context from Documents:
{context}

Instructions: Provide a comprehensive answer based only on the context above. Be precise, educational, and structure your response clearly."""

def generate_answer(prompt):
    if GEN_PIPE:
        try:
            # Check if it's Granite (text-generation) or FLAN-T5 (text2text-generation)
            if "granite" in GRANITE_MODEL.lower():
                # For Granite model
                out = GEN_PIPE(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, 
                             pad_token_id=GEN_PIPE.tokenizer.eos_token_id)
                if isinstance(out, list) and len(out) > 0:
                    generated = out[0].get("generated_text", "")
                    # Remove the original prompt from the generated text
                    if prompt in generated:
                        return generated.replace(prompt, "").strip()
                    return generated
            else:
                # For FLAN-T5 fallback
                out = GEN_PIPE(prompt, max_new_tokens=400)
                if isinstance(out, list):
                    return out[0].get("generated_text") or out[0].get("text") or str(out[0])
        except Exception as e:
            print("Pipeline error:", e)
    return "⚠ No generator available. Please check model loading."

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
        "Answers Saved": len(STATE["answers"]),
        "AI Model": GRANITE_MODEL if "granite" in str(GEN_PIPE) else "Fallback Model"
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
    gr.Markdown("# 📘 StudyMate — Powered by IBM Granite AI")
    gr.Markdown("*Advanced AI-powered study assistant using IBM Granite models for superior text generation*")

    with gr.Tab("Index PDFs"):  
        pdfs = gr.File(file_types=[".pdf"], file_count="multiple", type="filepath")  
        idx_btn = gr.Button("Index PDFs")  
        idx_out = gr.Textbox()  
        idx_btn.click(index_pdfs, pdfs, idx_out)  

    with gr.Tab("Q&A"):  
        q = gr.Textbox(label="Ask Question", placeholder="Enter your study question here...")  
        marks = gr.Radio([2,3,8,16], label="Marks", value=2)  
        go = gr.Button("Get Answer (Granite AI)")  
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

print("Launching StudyMate with IBM Granite AI... (share=True gives public link)")
demo.launch(share=True)
