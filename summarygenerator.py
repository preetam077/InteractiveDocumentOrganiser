# summarygenerator.py (Upgraded with Caching Logic)

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

# --- HELPER FUNCTIONS (Unchanged) ---

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

# --- MAIN CALLABLE FUNCTION (MODIFIED with Caching Logic) ---

def run_summary_generation(base_path_str: str):
    """
    Scans a directory, generates summaries for new files, and saves the combined output.
    Returns a dictionary containing a status message and KPI report.
    """
    base_path = Path(base_path_str)
    if not base_path.is_dir():
        return {"error": f"Provided path '{base_path_str}' is not a valid directory."}

    llm_output_file = 'llm_input.json'
    summary_cache = {}

    # --- CHANGE #1: Load existing summaries from the JSON file into a cache ---
    if os.path.exists(llm_output_file):
        try:
            with open(llm_output_file, 'r', encoding='utf-8') as f:
                existing_list = json.load(f)
                # Convert list of dicts to a dict keyed by file_path for fast lookups
                summary_cache = {item['file_path']: item for item in existing_list}
            print(f"Loaded {len(summary_cache)} existing summaries from cache.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load or parse {llm_output_file}. Starting fresh. Error: {e}")
            summary_cache = {}

    # KPI tracking variables
    processing_times: List[float] = []
    error_count: int = 0
    newly_processed_count: int = 0
    summary_lengths: List[int] = []
    cosine_similarities: List[float] = []

    # Step 1: Scan directory to get a current list of all files on disk
    print(f"Scanning {base_path} for all supported documents...")
    files_on_disk = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                files_on_disk.append(os.path.join(root, file))
    
    if not files_on_disk:
        return {"message": "No supported documents found in the specified directory.", "kpi_report": {}}

    # --- CHANGE #2: Prune cache and determine which files are new ---
    # Remove entries from cache if the file was deleted from disk
    existing_paths_on_disk = set(files_on_disk)
    cached_paths = set(summary_cache.keys())
    paths_to_remove = cached_paths - existing_paths_on_disk
    if paths_to_remove:
        print(f"\nPruning {len(paths_to_remove)} deleted file(s) from cache...")
        for path in paths_to_remove:
            del summary_cache[path]

    # Determine which files need to be processed
    files_to_process = [path for path in files_on_disk if path not in summary_cache]
    skipped_files_count = len(files_on_disk) - len(files_to_process)

    print(f"\nFound {len(files_on_disk)} total supported files.")
    if skipped_files_count > 0:
        print(f"---> Skipping {skipped_files_count} file(s) already in cache.")
    if not files_to_process:
        print("---> No new files to process.")
    else:
        print(f"---> Processing {len(files_to_process)} new file(s)...")


    # Step 2: Process only the new documents
    for file_path in files_to_process:
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

            # --- CHANGE #3: Add the new result directly to the cache dictionary ---
            summary_cache[file_path] = llm_dict
            newly_processed_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            error_count += 1
        finally:
            processing_times.append(time.time() - start_time)

    # Step 3: Save the updated, combined data for the LLM
    # --- CHANGE #4: Convert cache dictionary back to list before saving ---
    final_llm_input_data = list(summary_cache.values())
    try:
        with open(llm_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_llm_input_data, f, ensure_ascii=False, indent=4)
        print(f"\nOutput saved to {llm_output_file}. Total files in summary: {len(final_llm_input_data)}")
    except Exception as e:
        return {"error": f"Error saving {llm_output_file}: {str(e)}"}

    # Step 4: Prepare results to send back to the web UI
    total_files_in_summary = len(final_llm_input_data)
    kpi_report = {
        "files_found_on_disk": len(files_on_disk),
        "files_skipped_from_cache": skipped_files_count,
        "files_newly_processed": newly_processed_count,
        "total_files_in_summary": total_files_in_summary,
        "errors_in_this_run": error_count,
        "average_processing_time_for_new_files": (sum(processing_times) / len(processing_times)) if processing_times else 0.0
    }
    
    return {
        "message": f"Processing complete. Processed {newly_processed_count} new files. Total files in summary is now {total_files_in_summary}.",
        "kpi_report": kpi_report,
        "output_file": llm_output_file
    }