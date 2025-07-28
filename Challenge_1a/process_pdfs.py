import os
import json
import pdfplumber
import numpy as np
import pandas as pd
import joblib
from collections import Counter

# === Load Model and Label Encoder ===
model = joblib.load('./heading_classifier.pkl')
label_encoder = joblib.load('./label_encoder.pkl')


# === Feature Extraction from PDF ===
def extract_lines_from_pdf(pdf_path):
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_chars = page.chars
            if not page_chars:
                continue

            lines_dict = {}
            for char in page_chars:
                top_key = round(char['top'] / 3)
                if top_key not in lines_dict:
                    lines_dict[top_key] = []
                lines_dict[top_key].append(char)

            for top_key in sorted(lines_dict.keys()):
                chars = sorted(lines_dict[top_key], key=lambda c: c['x0'])
                line_text = ''.join(c['text'] for c in chars).strip()
                if not line_text or len(line_text) < 2:
                    continue

                fonts = [c['fontname'] for c in chars]
                font_sizes = [c['size'] for c in chars]
                dominant_font = Counter(fonts).most_common(1)[0][0]
                avg_font_size = round(np.mean(font_sizes), 2)
                is_bold = any('Bold' in f or 'bold' in f for f in fonts)
                is_italic = any('Italic' in f or 'Oblique' in f for f in fonts)
                uppercase_ratio = sum(1 for ch in line_text if ch.isupper()) / len(line_text)

                data.append({
                    'page': page_num ,
                    'text': line_text,
                    'font_size': avg_font_size,
                    'font_name': dominant_font,
                    'is_bold': is_bold,
                    'is_italic': is_italic,
                    'text_length': len(line_text),
                    'uppercase_ratio': round(uppercase_ratio, 3),
                    'x0': min(c['x0'] for c in chars),
                    'x1': max(c['x1'] for c in chars),
                    'y0': min(c['top'] for c in chars),
                    'y1': max(c['bottom'] for c in chars),
                })
    return pd.DataFrame(data)


# === Preprocess for Model Prediction ===
def preprocess_features(df):
    df['font_size_norm'] = df['font_size'] / df['font_size'].max()
    df['width'] = df['x1'] - df['x0']
    df['height'] = df['y1'] - df['y0']
    df['is_bold'] = df['is_bold'].astype(int)
    df['is_italic'] = df['is_italic'].astype(int)

    features = [
        'font_size_norm', 'text_length', 'uppercase_ratio',
        'is_bold', 'is_italic', 'x0', 'y0', 'width', 'height'
    ]
    return df, df[features]


# === Convert Predictions to JSON Format ===
def generate_json(df_with_preds):
    df_with_preds['predicted_label'] = df_with_preds['predicted_label'].str.upper()

    # Title (Page 1 TITLEs joined)
    title_parts = df_with_preds[
        (df_with_preds['page'] == 1) & (df_with_preds['predicted_label'] == 'TITLE')
    ]['text'].tolist()
    title = "  ".join(title_parts)

    # Outline entries
    outline = []
    for _, row in df_with_preds.iterrows():
        label = row['predicted_label']
        if label in ['H1', 'H2', 'H3']:
            outline.append({
                "level": label,
                "text": row['text'] + " ",
                "page": int(row['page'])
            })

    return {
        "title": title + "  ",
        "outline": outline
    }


# === Pipeline for One PDF ===
def process_pdf_to_json(pdf_path, output_path):
    print(f"📄 Processing {os.path.basename(pdf_path)}...")
    df = extract_lines_from_pdf(pdf_path)
    if df.empty:
        print("⚠️ No text found.")
        return

    df, X = preprocess_features(df)
    preds = model.predict(X)
    df['predicted_label'] = label_encoder.inverse_transform(preds)

    result_json = generate_json(df)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)
    print(f"✅ JSON saved: {output_path}")


# === Batch Process ===
def process_all_pdfs(input_dir="input", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        json_filename = os.path.splitext(pdf_file)[0] + ".json"
        output_path = os.path.join(output_dir, json_filename)
        process_pdf_to_json(pdf_path, output_path)


# === Run the Pipeline ===
if __name__ == "__main__":
    process_all_pdfs("input", "output")
