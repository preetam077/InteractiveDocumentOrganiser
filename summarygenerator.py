# summarygenerator.py

import os
import json
from typing import List, Dict
import pandas as pd
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import warnings
from torch.utils.data import dataloader
import time
from pathlib import Path

# --- ONE-TIME SETUP (runs when the server starts) ---

# Filter warnings
warnings.filterwarnings(
    "ignore",
    message="^Data Validation extension is not supported and will be removed$",
    category=UserWarning,
    module="openpyxl.worksheet._reader"
)
warnings.filterwarnings("ignore", 
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
    module=dataloader.__name__
)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt resource...")
    nltk.download('punkt')

# Load embedding model once
print("Loading sentence transformer model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Initialize document converter once
converter = DocumentConverter()

supported_extensions = {
    '.pdf', '.docx', '.pptx', '.xlsx', '.html',
    '.png', '.tiff', '.jpeg', '.jpg', '.gif', '.bmp',
    '.adoc', '.md', '.wav', '.mp3'
}

# --- HELPER FUNCTIONS ---

def generate_summary(text: str, doc_embedding: np.ndarray, summary_lengths: list, cosine_similarities: list, max_sentences: int = 5, is_table: bool = False) -> str:
    if not text.strip():
        return "No content available for summary."
    try:
        if is_table:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if not lines: return "No meaningful text content extracted from table."
            summary_lines = lines[:max_sentences]
            summary = f"Table summary: {'; '.join(summary_lines)}..."
            summary_lengths.append(len(summary.split()))
            return summary
        else:
            sentences = sent_tokenize(text)
            if not sentences: return "No sentences detected for summary."
            sentence_embeddings = embed_model.encode(sentences)
            similarities = cosine_similarity([doc_embedding], sentence_embeddings)[0]
            cosine_similarities.extend(similarities.tolist())
            top_indices = np.argsort(similarities)[-max_sentences:]
            top_sentences = [sentences[i] for i in sorted(top_indices) if similarities[i] > 0.1]
            summary = " ".join(top_sentences)
            summary_lengths.append(len(summary.split()))
            return summary if summary else "Unable to generate summary due to low similarity."
    except Exception as e:
        print(f"  -> Error generating summary: {str(e)}")
        return "Summary generation failed."

def extract_excel_text(file_path: str) -> str:
    try:
        xl = pd.ExcelFile(file_path)
        all_text = []
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name).fillna('').astype(str)
            headers = ", ".join(str(col) for col in df.columns if str(col).strip())
            all_text.append(f"Sheet: {sheet_name}\nHeaders: {headers}")
            for _, row in df.iterrows():
                row_text = ", ".join(str(val) for val in row if str(val).strip())
                if row_text: all_text.append(row_text)
        return "\n".join(all_text)
    except Exception as e:
        print(f"  -> Error reading Excel file {file_path} with pandas: {str(e)}")
        return ""

# --- MAIN CALLABLE FUNCTION ---

def run_summary_generation(base_path_str: str):
    """
    Scans a directory, generates summaries, and saves the output.
    Returns a dictionary containing a status message and KPI report.
    """
    base_path = Path(base_path_str)
    if not base_path.is_dir():
        return {"error": f"Provided path '{base_path_str}' is not a valid directory."}

    # KPI tracking variables
    supported_files: List[str] = []
    processing_times: List[float] = []
    error_count: int = 0
    successful_count: int = 0
    summary_lengths: List[int] = []
    cosine_similarities: List[float] = []
    output_files_success: List[str] = []
    output_data: List[Dict] = []
    llm_input_data: List[Dict] = []

    # Step 1: Scan directory
    print(f"Scanning {base_path} for documents...")
    for root, _, files in os.walk(base_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                supported_files.append(os.path.join(root, file))

    if not supported_files:
        return {"message": "No supported documents found in the specified directory.", "kpi_report": {}}

    # Step 2: Process documents
    print(f"\nProcessing {len(supported_files)} supported documents...")
    for file_path in supported_files:
        start_time = time.time()
        try:
            print(f"Processing: {file_path}")
            ext = os.path.splitext(file_path)[1].lower()
            llm_dict: Dict = {'file_path': file_path, 'type': ext, 'summary': ""}
            
            full_text = ""
            if ext == '.xlsx':
                full_text = extract_excel_text(file_path)
            else:
                conv_res = converter.convert(file_path)
                if conv_res.document:
                    doc: DoclingDocument = conv_res.document
                    text_parts = [item.text for item in doc.texts if item.text]
                    for table in doc.tables:
                        for row in table.data:
                            for cell in row:
                                cell_text = getattr(cell, 'text', str(cell)) if not isinstance(cell, str) else cell
                                if cell_text: text_parts.append(cell_text)
                    full_text = "\n".join(text_parts).strip()
            
            if full_text:
                doc_embedding = embed_model.encode(full_text)
                summary = generate_summary(full_text, doc_embedding, summary_lengths, cosine_similarities, is_table=(ext == '.xlsx'))
                llm_dict['summary'] = summary
            else:
                llm_dict['summary'] = "No content available for summary."

            llm_input_data.append(llm_dict)
            successful_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            error_count += 1
        finally:
            processing_times.append(time.time() - start_time)

    # Step 3: Save output for the LLM
    llm_output_file = 'llm_input.json'
    try:
        with open(llm_output_file, 'w', encoding='utf-8') as f:
            json.dump(llm_input_data, f, ensure_ascii=False, indent=4)
        print(f"Output saved to {llm_output_file}")
        output_files_success.append(llm_output_file)
    except Exception as e:
        return {"error": f"Error saving {llm_output_file}: {str(e)}"}

    # Step 4: Prepare results to send back to the web UI
    kpi_report = {
        "document_processing_success_rate": (successful_count / len(supported_files) * 100) if supported_files else 0.0,
        "average_processing_time_seconds": (sum(processing_times) / len(processing_times)) if processing_times else 0.0,
        "error_rate": (error_count / len(supported_files) * 100) if supported_files else 0.0,
        "files_found": len(supported_files),
        "files_processed_successfully": successful_count,
        "errors": error_count
    }
    
    return {
        "message": f"Processing complete. Successfully processed {successful_count} of {len(supported_files)} files.",
        "kpi_report": kpi_report,
        "output_file": llm_output_file
    }