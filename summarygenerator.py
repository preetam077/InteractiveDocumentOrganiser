# summarygenerator.py (Upgraded with Threading)

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
import concurrent.futures # --- NEW ---

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
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt resource...")
    nltk.download('punkt')

embed_model = None
converter = None

supported_extensions = {
    '.pdf', '.docx', '.pptx', '.xlsx', '.html',
    '.png', '.tiff', '.jpeg', '.jpg', '.gif', '.bmp',
    '.adoc', '.md', '.wav', '.mp3'
}

# --- HELPER FUNCTIONS (Unchanged) ---
def generate_summary(text: str, doc_embedding: np.ndarray, summary_lengths: list, cosine_similarities: list, max_sentences: int = 5, is_table: bool = False) -> str:
    # This function is being modified to be self-contained for threading
    # It now returns the summary and also the new lengths/similarities it generated
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
            sentence_embeddings = embed_model.encode(sentences)
            similarities = cosine_similarity([doc_embedding], sentence_embeddings)[0]
            top_indices = np.argsort(similarities)[-max_sentences:]
            top_sentences = [sentences[i] for i in sorted(top_indices) if similarities[i] > 0.1]
            summary = " ".join(top_sentences)
            summary_len = len(summary.split())
            final_summary = summary if summary else "Unable to generate summary due to low similarity."
            return final_summary, summary_len, similarities.tolist()
    except Exception as e:
        print(f"  -> Error generating summary: {str(e)}")
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
        print(f"  -> Error reading Excel file {file_path} with pandas: {str(e)}")
        return ""

# --- NEW: WORKER FUNCTION FOR A SINGLE FILE ---
def process_single_file(file_path: str) -> Dict[str, Any]:
    """
    This function contains all the logic to process one file.
    It's designed to be run in a separate thread.
    It returns a dictionary with the results for thread-safe collection.
    """
    global embed_model, converter
    if embed_model is None:
        print(f"Initializing models in process ID: {os.getpid()}")
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        converter = DocumentConverter()
        print(f"Models loaded for process ID: {os.getpid()}")

    start_time = time.time()
    try:
        print(f"Processing: {file_path} in process {os.getpid()}")
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
        
        summary_len = 0
        similarities = []
        if full_text:
            doc_embedding = embed_model.encode(full_text)
            # Dummy lists that won't be used, as we capture the return values
            summary, summary_len, similarities = generate_summary(
                full_text, doc_embedding, [], [], is_table=(ext == '.xlsx')
            )
            llm_dict['summary'] = summary
        else:
            llm_dict['summary'] = "No content available for summary."

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

# --- MAIN CALLABLE FUNCTION (MODIFIED with Threading) ---
def run_summary_generation(base_path_str: str, max_workers: int = None): # Added max_workers parameter
    """
    Scans a directory, generates summaries for new files using multiple threads, and saves the output.
    Returns a dictionary containing a status message and KPI report.
    """
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

    # Step 1: Scan and determine files to process (Unchanged logic)
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

    # --- MODIFIED: Process new documents using a ProcessPoolExecutor ---
    if files_to_process:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks to the executor
            future_to_file = {executor.submit(process_single_file, file_path): file_path for file_path in files_to_process}
            
            # Process results as they are completed
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                
                # This part is now thread-safe because we are processing results one by one in the main thread
                processing_times.append(result["processing_time"])
                
                if result["status"] == "success":
                    llm_dict = result["data"]
                    summary_cache[llm_dict['file_path']] = llm_dict
                    summary_lengths.append(result["summary_length"])
                    cosine_similarities.extend(result["cosine_similarities"])
                    newly_processed_count += 1
                else: # status == "error"
                    error_count += 1
    
    # --- Step 3 & 4: Saving and Reporting (Unchanged Logic) ---
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