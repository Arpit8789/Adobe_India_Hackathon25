import os
import json
import faiss
import numpy as np
from datetime import datetime
from typing import List
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Constants
COLLECTIONS = ["Collection1", "Collection2", "Collection3"]
MODEL_PATH = "./models/all-MiniLM-L6-v2"
OUTPUT_ROOT = "output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Load model once
model = SentenceTransformer(MODEL_PATH)

def estimate_tokens(text: str) -> int:
    return len(text.split())

def extract_blocks_from_pdf(pdf_path: str) -> List[dict]:
    blocks = []
    for page_layout in extract_pages(pdf_path):
        page_number = page_layout.pageid
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if not text:
                    continue
                font_sizes = [char.size for line in element for char in line if isinstance(char, LTChar)]
                avg_font_size = round(sum(font_sizes) / len(font_sizes), 2) if font_sizes else 0
                blocks.append({
                    "text": text,
                    "font_size": avg_font_size,
                    "page_number": page_number
                })
    return blocks

def chunk_blocks(blocks: List[dict], max_tokens: int = 350) -> List[dict]:
    chunks = []
    current_chunk, current_tokens = [], 0
    for block in blocks:
        tokens = estimate_tokens(block["text"])
        if tokens > max_tokens:
            for sentence in block["text"].split('. '):
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_tokens = estimate_tokens(sentence)
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append({
                        "chunk_text": " ".join(b["text"] for b in current_chunk),
                        "page_numbers": list({b["page_number"] for b in current_chunk})
                    })
                    current_chunk, current_tokens = [], 0
                current_chunk.append({"text": sentence, "page_number": block["page_number"]})
                current_tokens += sentence_tokens
            continue

        if current_tokens + tokens > max_tokens and current_chunk:
            chunks.append({
                "chunk_text": " ".join(b["text"] for b in current_chunk),
                "page_numbers": list({b["page_number"] for b in current_chunk})
            })
            current_chunk, current_tokens = [], 0

        current_chunk.append(block)
        current_tokens += tokens

    if current_chunk:
        chunks.append({
            "chunk_text": " ".join(b["text"] for b in current_chunk),
            "page_numbers": list({b["page_number"] for b in current_chunk})
        })

    return chunks

def process_collection(collection_name: str):
    print(f"\n🔍 Processing {collection_name}")
    base_path = os.path.join(collection_name)
    pdf_dir = os.path.join(base_path, "PDFs")
    input_json_path = os.path.join(base_path, "input.json")
    chunked_dir = os.path.join(base_path, "chunked_output")
    index_dir = os.path.join(base_path, "faiss_index")
    os.makedirs(chunked_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    # Step 1: Chunk PDFs
    for filename in tqdm(os.listdir(pdf_dir), desc=f"📄 Chunking PDFs in {collection_name}"):
        if not filename.lower().endswith(".pdf"):
            continue
        blocks = extract_blocks_from_pdf(os.path.join(pdf_dir, filename))
        blocks.sort(key=lambda b: (b["page_number"], -b["font_size"]))
        chunks = chunk_blocks(blocks)
        for chunk in chunks:
            chunk["filename"] = filename
        with open(os.path.join(chunked_dir, f"{os.path.splitext(filename)[0]}.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Step 2: Build FAISS index
    embeddings, metadata = [], []
    for fname in os.listdir(chunked_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(chunked_dir, fname), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for chunk in chunks:
            text = chunk["chunk_text"].strip()
            if not text:
                continue
            emb = model.encode(text)
            embeddings.append(emb)
            metadata.append({
                "filename": chunk["filename"],
                "page_numbers": chunk["page_numbers"],
                "chunk_text": text
            })
    if not embeddings:
        raise ValueError("No embeddings found.")
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Step 3: Retrieval
    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    docs = [doc["filename"] for doc in input_data["documents"]]
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]

    query = f"You are a {persona}. Your goal is to: {job}"
    query_embedding = model.encode(query).astype("float32").reshape(1, -1)
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    distances, indices = index.search(query_embedding, 10)
    extracted_sections, subsection_analysis, seen, rank = [], [], set(), 1

    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx == -1 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        filename = chunk["filename"]
        if filename not in docs:
            continue
        page_numbers = chunk.get("page_numbers", [-1])
        doc_key = (filename, page_numbers[0])
        if doc_key in seen:
            continue
        seen.add(doc_key)
        chunk_text = chunk["chunk_text"]
        section_title = chunk_text.split("\n")[0][:100]
        extracted_sections.append({
            "document": filename,
            "section_title": section_title,
            "importance_rank": rank,
            "page_number": page_numbers[0]
        })
        subsection_analysis.append({
            "document": filename,
            "refined_text": chunk_text.strip(),
            "page_number": page_numbers[0]
        })
        rank += 1
        if rank > 5:
            break

    # Step 4: Save output
    output_data = {
        "metadata": {
            "input_documents": docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    output_path = os.path.join(OUTPUT_ROOT, f"{collection_name.lower()}_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Output written to {output_path}")

# Run for all collections
if __name__ == "__main__":
    for col in COLLECTIONS:
        process_collection(col)
