import os
import json
from typing import List
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
from tqdm import tqdm

input_dir = "input"
output_dir = "chunked_output"
os.makedirs(output_dir, exist_ok=True)

# Estimate token count using simple split (fallback)
def estimate_tokens(text: str) -> int:
    return len(text.split())

# Extract text blocks with font size and page number
def extract_blocks_from_pdf(pdf_path: str) -> List[dict]:
    blocks = []
    for page_layout in extract_pages(pdf_path):
        page_number = page_layout.pageid
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if not text:
                    continue

                font_sizes = []
                for text_line in element:
                    for char in text_line:
                        if isinstance(char, LTChar):
                            font_sizes.append(char.size)
                avg_font_size = round(sum(font_sizes) / len(font_sizes), 2) if font_sizes else 0

                blocks.append({
                    "text": text,
                    "font_size": avg_font_size,
                    "page_number": page_number
                })
    return blocks

# Chunk blocks based on max token count
def chunk_blocks(blocks: List[dict], max_tokens: int = 350) -> List[dict]:
    chunks = []
    current_chunk = []
    current_tokens = 0

    for block in blocks:
        tokens = estimate_tokens(block["text"])

        # If block itself is too large, split by sentence
        if tokens > max_tokens:
            sentences = block["text"].split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_tokens = estimate_tokens(sentence)
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    combined_text = " ".join(b["text"] for b in current_chunk)
                    chunks.append({
                        "chunk_text": combined_text,
                        "page_numbers": list(sorted(set(b["page_number"] for b in current_chunk)))
                    })
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append({
                    "text": sentence,
                    "page_number": block["page_number"]
                })
                current_tokens += sentence_tokens
            continue

        # Normal chunking
        if current_tokens + tokens > max_tokens and current_chunk:
            combined_text = " ".join(b["text"] for b in current_chunk)
            chunks.append({
                "chunk_text": combined_text,
                "page_numbers": list(sorted(set(b["page_number"] for b in current_chunk)))
            })
            current_chunk = []
            current_tokens = 0

        current_chunk.append(block)
        current_tokens += tokens

    # Add last chunk
    if current_chunk:
        combined_text = " ".join(b["text"] for b in current_chunk)
        chunks.append({
            "chunk_text": combined_text,
            "page_numbers": list(sorted(set(b["page_number"] for b in current_chunk)))
        })

    return chunks

# Main PDF processing loop
for filename in tqdm(os.listdir(input_dir), desc="Chunking PDFs"):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_dir, filename)
    try:
        blocks = extract_blocks_from_pdf(pdf_path)
        # Optional: sort blocks to ensure consistent order
        blocks.sort(key=lambda b: (b["page_number"], -b["font_size"]))
        chunks = chunk_blocks(blocks)

        for chunk in chunks:
            chunk["filename"] = filename  # add filename to each chunk

        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        continue

print(f"✅ Done chunking PDFs into '{output_dir}'")


import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Directories
input_dir = "chunked_output"
output_dir = "faiss_index"
os.makedirs(output_dir, exist_ok=True)

# Load sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Containers
all_embeddings = []
metadata = []

print(f"🔍 Reading JSON files from '{input_dir}'...")
for filename in tqdm(os.listdir(input_dir), desc="📄 Processing files"):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(input_dir, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error reading {filename}: {e}")
        continue

    for chunk in chunks:
        chunk_text = chunk.get("chunk_text", "").strip()
        if not chunk_text:
            continue

        embedding = model.encode(chunk_text, show_progress_bar=False)
        all_embeddings.append(embedding)

        metadata.append({
            "filename": chunk.get("filename", filename),
            "page_numbers": chunk.get("page_numbers", [-1]),
            "chunk_text": chunk_text
        })

# Fail-safe
if not all_embeddings:
    raise ValueError("🚫 No valid text chunks found in 'chunked_output'.")

# Convert to FAISS index
print(f"\n💾 Building FAISS index with {len(all_embeddings)} vectors...")
embedding_dim = len(all_embeddings[0])
all_embeddings_np = np.array(all_embeddings).astype("float32")

# Optional normalization (comment if not desired)
# faiss.normalize_L2(all_embeddings_np)

index = faiss.IndexFlatL2(embedding_dim)
index.add(all_embeddings_np)

# Save FAISS index
faiss_index_path = os.path.join(output_dir, "index.faiss")
faiss.write_index(index, faiss_index_path)

# Save metadata
metadata_path = os.path.join(output_dir, "index_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ FAISS index saved at: {faiss_index_path}")
print(f"🗃️ Metadata saved at: {metadata_path}")


import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Paths
INPUT_JSON = "./input/input.json"
FAISS_INDEX_PATH = "faiss_index/index.faiss"
METADATA_PATH = "faiss_index/index_metadata.json"
OUTPUT_JSON = "./output/output.json"

# Load FAISS index
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"🚫 FAISS index not found at {FAISS_INDEX_PATH}")
index = faiss.read_index(FAISS_INDEX_PATH)

# Load metadata
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"🚫 Metadata file not found at {METADATA_PATH}")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load SentenceTransformer model
model = SentenceTransformer("./models/all-MiniLM-L6-v2")


# Load input data
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    input_data = json.load(f)

documents = [doc["filename"] for doc in input_data["documents"]]
persona = input_data["persona"]["role"]
job = input_data["job_to_be_done"]["task"]

# Step 1: Formulate query
query = f"You are a {persona}. Your goal is to: {job}"
query_embedding = model.encode(query).astype('float32').reshape(1, -1)

# Step 2: Search FAISS index
top_k = 10  # can be increased if needed
distances, indices = index.search(query_embedding, top_k)

# Step 3: Extract relevant sections and refined text
extracted_sections = []
subsection_analysis = []

seen = set()
rank = 1

for i in range(len(indices[0])):
    idx = indices[0][i]
    if idx == -1 or idx >= len(metadata):
        continue

    chunk = metadata[idx]
    filename = chunk.get("filename")
    page_numbers = chunk.get("page_numbers", [-1])
    chunk_text = chunk.get("chunk_text", "")

    if filename not in documents:
        continue  # Skip unrelated files

    doc_key = (filename, page_numbers[0])
    if doc_key in seen:
        continue  # Skip duplicates
    seen.add(doc_key)

    # First part: extracted_sections
    section_title = chunk_text.strip().split("\n")[0][:100]  # Heuristic for section title
    extracted_sections.append({
        "document": filename,
        "section_title": section_title,
        "importance_rank": rank,
        "page_number": page_numbers[0]
    })

    # Second part: subsection_analysis
    subsection_analysis.append({
        "document": filename,
        "refined_text": chunk_text.strip(),
        "page_number": page_numbers[0]
    })

    rank += 1
    if rank > 5:
        break  # Only top 5 for output

# Step 4: Build output JSON
output_data = {
    "metadata": {
        "input_documents": documents,
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.now().isoformat()
    },
    "extracted_sections": extracted_sections,
    "subsection_analysis": subsection_analysis
}

# Step 5: Save output
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ Output saved to '{OUTPUT_JSON}'")
