# summarygenerator.py (Upgraded with Hybrid Extraction)

import os
import json
from typing import List, Dict, Tuple, Any
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
from textwrap import wrap
import fitz  # --- NEW: Import PyMuPDF ---
import concurrent.futures

# --- ONE-TIME SETUP (Unchanged) ---
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

embed_model = None
converter = None
def worker_init():
    """
    Initializer function for each worker process.
    Loads the ML models into the global scope of the worker.
    """
    global embed_model, converter
    print(f"Initializing models in process ID: {os.getpid()}")
    embed_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    converter = DocumentConverter()
    print(f"Models loaded for process ID: {os.getpid()}")

supported_extensions = {
    '.pdf', '.docx', '.pptx', '.xlsx', '.html',
    '.png', '.tiff', '.jpeg', '.jpg', '.gif', '.bmp',
    '.adoc', '.md', '.wav', '.mp3'
}

# --- HELPER FUNCTIONS ---
def generate_summary(text: str, doc_embedding: np.ndarray, summary_lengths: list, cosine_similarities: list, max_sentences: int = 5, is_table: bool = False) -> Tuple[str, int, list]:
    if not text.strip():
        return "No content available for summary.", 0, []
    try:
        if is_table:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if not lines: return "No meaningful text content extracted from table.", 0, []
            summary_lines = lines[:max_sentences]
            summary = f"Table summary: {'; '.join(summary_lines)}..."
            summary_len = len(summary.split())
            return summary, summary_len, []
        else:
            sentences = sent_tokenize(text)
            if not sentences: return "No sentences detected for summary.", 0, []
            sentence_embeddings = embed_model.encode(sentences, batch_size=32, show_progress_bar=False)
            similarities = cosine_similarity([doc_embedding], sentence_embeddings)[0]
            top_indices = np.argsort(similarities)[-max_sentences:]
            top_sentences = [sentences[i] for i in sorted(top_indices) if similarities[i] > 0.1]
            summary = " ".join(top_sentences)
            summary_len = len(summary.split())
            final_summary = summary if summary else "Unable to generate summary due to low similarity."
            return final_summary, summary_len, similarities.tolist()
    except Exception as e:
        print(f"   -> Error generating summary: {str(e)}")
        return "Summary generation failed.", 0, []

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
        print(f"   -> Error reading Excel file {file_path} with pandas: {str(e)}")
        return ""

# --- NEW: HYBRID TEXT EXTRACTOR ---
def extract_text_hybrid(file_path: str, file_ext: str) -> str:
    """
    Extracts text using a smart hybrid approach.
    - Tries the fast PyMuPDF for PDFs.
    - If PyMuPDF returns very little text (indicating a scanned PDF),
      it automatically falls back to the docling converter for OCR.
    - Uses docling for all other file types.
    """
    print(f" -> Running hybrid extractor for '{os.path.basename(file_path)}'...")
    if file_ext == '.pdf':
        try:
            # --- PLAN A: Try the fast PyMuPDF method first ---
            with fitz.open(file_path) as doc:
                text_parts = [page.get_text() for page in doc]
                full_text = "\n".join(text_parts).strip()
                
                # --- SMART DETECTION LOGIC ---
                # If the text is very short (e.g., less than 20 chars per page),
                # assume it's a scanned PDF and needs OCR.
                if len(full_text) < (len(doc) * 20):
                    print("   -> PyMuPDF found little text. Assuming scanned PDF, falling back to docling for OCR.")
                    raise ValueError("Potential scanned document detected.") # Intentionally trigger fallback
            
            return full_text # Return the fast result if it was good
        except Exception as e:
            # --- PLAN B: Fallback to docling for OCR or on any error ---
            print(f"   -> PyMuPDF failed or fallback triggered: {e}. Using docling converter.")
            conv_res = converter.convert(file_path)
            if conv_res.document:
                return "\n".join([item.text for item in conv_res.document.texts if item.text]).strip()
            return ""

    elif file_ext == '.xlsx':
        return extract_excel_text(file_path)
    
    else: # For .docx, .pptx, and all others, use the original docling converter
        conv_res = converter.convert(file_path)
        if conv_res.document:
            doc: DoclingDocument = conv_res.document
            text_parts = [item.text for item in doc.texts if item.text]
            for table in doc.tables:
                for row in table.data:
                    for cell in row:
                        cell_text = getattr(cell, 'text', str(cell))
                        if cell_text: text_parts.append(cell_text)
            return "\n".join(text_parts).strip()
    return ""

# --- MODIFIED: WORKER FUNCTION FOR A SINGLE FILE ---
def process_single_file(file_path: str) -> Dict[str, Any]:
    """
    Processes a single file using the hybrid text extractor and adaptive summarization.
    """
    CHUNKING_WORD_THRESHOLD = 2000
    start_time = time.time()
    try:
        print(f"Processing: {os.path.basename(file_path)} in process {os.getpid()}")
        ext = os.path.splitext(file_path)[1].lower()
        llm_dict: Dict = {'file_path': file_path, 'type': ext, 'summary': ""}
        
        # --- MODIFIED: Use the new hybrid extractor function ---
        full_text = extract_text_hybrid(file_path, ext)
        
        summary_len = 0
        similarities = []
        if not full_text:
            llm_dict['summary'] = "No content available for summary."
        # --- ADAPTIVE LOGIC (Unchanged) ---
        elif len(full_text.split()) > CHUNKING_WORD_THRESHOLD:
            # --- PATH 1: Large Document -> Use Hierarchical Chunking ---
            print(f" -> Large document ({len(full_text.split())} words). Applying hierarchical summary...")
            chunks = wrap(full_text, width=8000, break_long_words=False, replace_whitespace=False)
            intermediate_summaries = []
            for chunk in chunks:
                if not chunk.strip(): continue
                chunk_embedding = embed_model.encode(chunk)
                intermediate_summary, _, _ = generate_summary(
                    chunk, chunk_embedding, [], [], max_sentences=2 
                )
                if "generation failed" not in intermediate_summary and "No content" not in intermediate_summary:
                    intermediate_summaries.append(intermediate_summary)
            
            final_text_to_summarize = " ".join(intermediate_summaries)
            if final_text_to_summarize:
                final_embedding = embed_model.encode(final_text_to_summarize)
                summary, summary_len, similarities = generate_summary(
                    final_text_to_summarize, final_embedding, [], [], max_sentences=5
                )
                llm_dict['summary'] = summary
            else:
                llm_dict['summary'] = "Failed to generate intermediate summaries for chunked document."
        else:
            # --- PATH 2: Small/Medium Document -> Use Original Method ---
            print(f" -> Standard document ({len(full_text.split())} words). Processing directly.")
            doc_embedding = embed_model.encode(full_text)
            summary, summary_len, similarities = generate_summary(
                full_text, doc_embedding, [], [], is_table=(ext == '.xlsx')
            )
            llm_dict['summary'] = summary

        return {
            "status": "success",
            "data": llm_dict,
            "processing_time": time.time() - start_time,
            "summary_length": summary_len,
            "cosine_similarities": similarities
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_path": file_path,
            "error_message": str(e),
            "processing_time": time.time() - start_time
        }

# --- MAIN CALLABLE FUNCTION (Unchanged) ---
def run_summary_generation(base_path_str: str, max_workers: int = None):
    # ... (The rest of your code is unchanged) ...
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt resource...")
        nltk.download('punkt')

    if max_workers is None:
        max_workers = os.cpu_count() or 4
        print(f"--- Auto-detected {max_workers} CPU cores to use as workers. ---")
        
    base_path = Path(base_path_str)
    if not base_path.is_dir():
        return {"error": f"Provided path '{base_path_str}' is not a valid directory."}

    llm_output_file = 'llm_input.json'
    summary_cache = {}

    if os.path.exists(llm_output_file):
        try:
            with open(llm_output_file, 'r', encoding='utf-8') as f:
                existing_list = json.load(f)
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

    # Step 1: Scan and determine files to process
    print(f"Scanning {base_path} for all supported documents...")
    files_on_disk = [os.path.join(root, file) for root, _, files in os.walk(base_path) for file in files if os.path.splitext(file)[1].lower() in supported_extensions]
    if not files_on_disk:
        return {"message": "No supported documents found in the specified directory.", "kpi_report": {}}
    
    existing_paths_on_disk = set(files_on_disk)
    cached_paths = set(summary_cache.keys())
    paths_to_remove = cached_paths - existing_paths_on_disk
    if paths_to_remove:
        print(f"\nPruning {len(paths_to_remove)} deleted file(s) from cache...")
        for path in paths_to_remove:
            del summary_cache[path]
    
    files_to_process = [path for path in files_on_disk if path not in summary_cache]
    skipped_files_count = len(files_on_disk) - len(files_to_process)

    print(f"\nFound {len(files_on_disk)} total supported files.")
    if skipped_files_count > 0:
        print(f"---> Skipping {skipped_files_count} file(s) already in cache.")
    if not files_to_process:
        print("---> No new files to process.")
    else:
        print(f"---> Processing {len(files_to_process)} new file(s) using up to {max_workers} threads...")

    # Process new documents using a ProcessPoolExecutor
    if files_to_process:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers,initializer=worker_init) as executor:
            future_to_file = {executor.submit(process_single_file, file_path): file_path for file_path in files_to_process}
            
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                
                processing_times.append(result["processing_time"])
                
                if result["status"] == "success":
                    llm_dict = result["data"]
                    summary_cache[llm_dict['file_path']] = llm_dict
                    summary_lengths.append(result["summary_length"])
                    cosine_similarities.extend(result["cosine_similarities"])
                    newly_processed_count += 1
                else: # status == "error"
                    error_count += 1
    
    # Saving and Reporting
    final_llm_input_data = list(summary_cache.values())
    try:
        with open(llm_output_file, 'w', encoding='utf-8') as f:
            json.dump(final_llm_input_data, f, ensure_ascii=False, indent=4)
        print(f"\nOutput saved to {llm_output_file}. Total files in summary: {len(final_llm_input_data)}")
    except Exception as e:
        return {"error": f"Error saving {llm_output_file}: {str(e)}"}

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