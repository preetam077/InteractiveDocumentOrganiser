# fileorganizer.py (Final Version)
# This script has been updated to exclusively use the client.models.generate_content pattern
# and includes the fix for the tool-calling error.

import json
import os
import shutil
from google import genai
from dotenv import load_dotenv
from pathlib import Path
from google.genai import types # Required for GenerateContentConfig
from flask import session

# --- ONE-TIME SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the client as requested.
try:
    if not GOOGLE_API_KEY:
        raise ValueError("Error: GOOGLE_API_KEY environment variable is not set.")
    client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"CRITICAL ERROR initializing Google GenAI: {e}")
    client = None # Ensure client is None if initialization fails

# --- HELPER FUNCTIONS (UNCHANGED) ---

def load_document_data(filepath='llm_input.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading document data: {e}")
        return None

def create_file_path_map(document_data):
    path_map = {}
    for doc in document_data:
        filename = os.path.basename(doc['file_path'])
        path_map[filename] = doc['file_path']
    return path_map

def search_documents(all_docs: list, query: str):
    """
    Searches document summaries for a specific query string.
    Only returns documents where the query is found in the summary.
    """
    # ADD THIS LINE FOR DEBUGGING: See the exact query the AI is using.
    print(f"\n---> AI generated query: '{query}'")
    
    print(f"---> TOOL EXECUTED: Searching for documents containing '{query}'...")
    if not all_docs:
        return {"error": "Document data is not available for search."}
    matches = []
    for doc in all_docs:
        if query.lower() in doc.get("summary", "").lower():
            matches.append({"file_path": doc.get("file_path"), "summary": doc.get("summary")})
    print(f"---> TOOL: Found {len(matches)} matching documents.")
    return {"found_documents": matches}


# --- MAIN CALLABLE FUNCTIONS (UPDATED) ---

def get_initial_analysis():
    """Loads data from llm_input.json and gets the initial AI analysis."""
    if not client:
        return {"error": "Google GenAI is not initialized. Check API key."}
        
    all_docs = load_document_data()
    if not all_docs:
        return {"error": "Could not load llm_input.json. Run the summary generator first."}

    documents_str = "\n".join([
        f"- File Path: {doc['file_path']}\n  Type: {doc.get('type', 'N/A')}\n  Summary: {doc['summary']}\n"
        for doc in all_docs
    ])
    prompt = f"""
    You are an expert file organization assistant. Your task is to analyze the current file structure based on the provided file paths, types, and summaries, and explain why the existing structure may not be optimal.

    **File Information:**
    {documents_str}

    **Instructions:**
    Respond ONLY with a concise analysis (200-400 words) that includes:
    - A description of the current structure.
    - Why it may not be optimal (e.g., scattered files, lack of logical grouping).
    - Suggestions for improvement (high-level).
    - A convincing argument on the benefits of reorganizing.

    Do not provide a JSON plan or file tree yet. Focus on analysis and persuasion.
    """
    try:
        model_name = 'gemini-2.5-flash-lite' # Using a reliable current model
        response = client.models.generate_content(
            model=model_name, contents=prompt
        )
        analysis_text = response.text.strip()
        return {"analysis": analysis_text, "all_docs": all_docs}
    except Exception as e:
        return {"error": f"Failed to get analysis from AI: {e}"}

# client = ... (your initialized Google GenAI client)
# from google.generativeai import types # Make sure this is imported

def answer_a_question(question: str, all_docs: list, current_analysis: str):
    """
    Answers a user question using an advanced two-call 'search then reason' approach.
    1. First call to the AI decides if a search is needed and generates a BROAD query.
    2. The tool runs a simple keyword search.
    3. Second call to the AI intelligently filters the results to give a precise answer.
    """
    if not client:
        return {"error": "Google GenAI is not initialized."}

    # This is the tool the AI will be able to call.
    # Its docstring is crucial for the AI to understand how to use it.
    def search_for_document_info(query: str):
        """
        Performs a broad keyword search to get a list of potentially relevant documents.
        This tool is the FIRST STEP. It retrieves documents based on keywords only.
        The final filtering and comparison (e.g., for numbers, dates) must be done
        by you, the assistant, after this tool returns its results.
        For example, for 'resumes with less than 7 years experience', a good query
        is simply 'experience'.
        """
        # This calls your actual search tool function
        return search_documents(all_docs=all_docs, query=query)

    try:
        model_name = 'gemini-2.5-flash-lite' # Per your instructions

        # --- PROMPT 1: For Router Logic and Broad Query Generation ---
        # This prompt instructs the model to either answer from context or
        # create a SIMPLE keyword query suitable for our "dumb" tool.
        prompt_1_search_generation = f"""
        You are a highly advanced file assistant. Your goal is to answer the user's question about a set of documents based on the context provided with their question.

        **Core Directive: You must operate autonomously. Follow the entire process below without pausing to ask the user for permission to proceed. Your only output should be the final answer.**
        **Your Reasoning Process is a mandatory two-stage process:**

        **Stage 1: Analyze User Intent and Decide Action**
        1.  **Analyze User Intent:** Carefully examine the user's latest message in the context of the entire conversation. Determine the user's primary intent by choosing one of the following two categories:
            A. **New Search Request:**  The user wants to find documents they have not asked for before. This usually involves new topics, keywords, or criteria (e.g., "Find me files about marketing," "Get all resumes with Python experience," "now show me documents about finance").
            B. **Conversational Follow-up:** The user is asking a question, making a statement, or offering a correction about the documents you presented in your immediately preceding response (e.g., "Tell me more about the second file," "Which of these is most recent?", "I think that first document is incorrect.").
        2.  **Decide on an Action:**
            - If the intent is a New Search Request (A): You MUST use the `search_for_document_info` tool to find a new list of documents. Simplify the user's request to a single, broad keyword for the tool (e.g., for 'resumes with >5 years experience', the query is 'experience').
            - If the intent is a Conversational Follow-up (B): You MUST NOT use the search tool again. You will answer using only the information (file paths and summaries) you already have from the previous turn.
        
        **Stage 2: Formulate and Provide the Answer**
        3.  **Filter & Synthesize (CRUCIAL):** This is your most important task.
            - For New Searches: After getting results from the tool, act as an intelligent filter. Meticulously review the summaries and compare them against the user's original, specific request. Perform necessary comparisons (e.g., checking if '8 years' is 'more than 5 years').
            - For Conversational Follow-ups: Re-examine the summaries from the previous results in light of the user's new question or correction
        4.  **Answer:** 
            - If answering a New Search: Provide a direct and concise list of matching documents. For each document, list its file path and a brief explanation of its relevance based on its summary. If no documents match your filtering, state that.
            - If answering a Follow-up: Directly address the user's question or statement. For example, if they challenge a file's content, respond with "You asked about [file name]. Based on the summary I have, it appears to be about [topic from summary], and I do not see any information related to [challenged topic]."

        **Context from Initial Analysis:**
        "{current_analysis}"
        """

        # --- CALL 1: Decide whether to search and get the broad query ---

        chat_history_dicts = session.get('chat_history', [])
        
        if not chat_history_dicts:
            chat_history_dicts.append({'role': 'user', 'parts': [{'text': prompt_1_search_generation.strip()}]})
            chat_history_dicts.append({'role': 'model', 'parts': [{"text":"Understood. I am ready to answer questions."}]})

        print(f"Restored chat history with {len(chat_history_dicts)} entries.")

        # Convert history from dicts to 'Content' objects for the API
        chat_history = [types.Content(**item) for item in chat_history_dicts]

        print(f"---> Sending to AI (Call 1) with {len(chat_history)} history entries...")

        chat = client.chats.create(
            model=model_name,
            history=chat_history,
            config=types.GenerateContentConfig(system_instruction=prompt_1_search_generation,
                                               tools=[search_for_document_info],)
            )
        response = chat.send_message(question)

        def content_to_dict(content):
            """Convert a single Content object to a dictionary for JSON serialization."""
            parts = []
            for part in content.parts:
                part_dict = {}
                # Handle text content
                if part.text is not None:
                    part_dict["text"] = part.text
                # Add other fields as needed (e.g., blob data, inline_data)
                if part_dict:  # Only add non-empty part dictionaries
                    parts.append(part_dict)
            return {
                "role": content.role,
                "parts": parts
            }

        updated_history_dicts = [content_to_dict(content) for content in chat.get_history()]
        session['chat_history'] = updated_history_dicts

        return {"answer": response.text}

    except Exception as e:
        print(f"An exception occurred in answer_a_question: {e}")
        return {"error": f"Failed to get an answer from the AI: {str(e)}"}


def get_organization_plan(all_docs, current_analysis):
    """Gets the file organization plan, tree, and reasoning from the AI."""
    if not client:
        return {"error": "Google GenAI is not initialized."}
        
    documents_str = "\n".join([
        f"- File: {os.path.basename(doc['file_path'])}\n  Summary: {doc['summary']}\n"
        for doc in all_docs
    ])
    prompt = f"""
    You are an expert file organization assistant. Based on the following analysis, your task is to organize the files into a logical folder structure and provide a JSON plan, an ASCII file tree, and your reasoning.DO NOT CHANGE THE NAME OF FILES

    **Previous Analysis:**
    {current_analysis}

    **File Information:**
    {documents_str}

    **Instructions:**
    Respond ONLY with a single output containing three clearly separated sections (JSON, file tree, reasoning). DO NOT CHANGE THE FILENAMES IN ANY CASE.

    **Output Format**:
    ```json
    {{
      "Project_A/Source_Code": ["main.py", "utils.py"]
    }}
    -----
    Project_A/
    └── Source_Code/
        ├── main.py
        └── utils.py
    -----
    The reasoning for this structure is...
    ```
    """
    try:
        # Using a model better suited for complex reasoning and formatting.
        model_name = 'gemini-2.5-flash-lite'
        response = client.models.generate_content(
            model=model_name, contents=prompt
        )
        response_text = response.text.strip().replace('```json', '').replace('```', '')
        parts = response_text.split('-----', 2)
        if len(parts) != 3:
            raise ValueError("Invalid response format from AI. Could not split into 3 parts.")
        
        json_part, file_tree, reasoning = [part.strip() for part in parts]
        plan = json.loads(json_part)
        return {"plan": plan, "file_tree": file_tree, "reasoning": reasoning}
    except Exception as e:
        return {"error": f"Failed to get a valid plan from the AI: {e}"}


def execute_the_plan(plan, all_docs, destination_root_str):
    """Executes the file moving plan. Does not call the AI."""
    destination_root = Path(destination_root_str)
    if not destination_root.is_dir():
        try:
            os.makedirs(destination_root)
        except Exception as e:
            return {"error": f"Destination root '{destination_root_str}' does not exist and could not be created: {e}"}
            
    file_path_map = create_file_path_map(all_docs)
    log = []
    files_moved = 0
    errors_encountered = 0
    
    for directory, filenames in plan.items():
        new_dir_path = destination_root / Path(directory)
        try:
            os.makedirs(new_dir_path, exist_ok=True)
            log.append(f"[OK] Ensured directory exists: '{new_dir_path}'")
        except OSError as e:
            log.append(f"[ERROR] Could not create directory '{new_dir_path}'. Error: {e}")
            errors_encountered += 1
            continue
            
        for filename in filenames:
            original_path_str = file_path_map.get(filename)
            if not original_path_str:
                log.append(f"  [WARN] Could not find original path for '{filename}'. Skipping.")
                errors_encountered += 1
                continue
                
            original_path = Path(original_path_str)
            if not original_path.exists():
                log.append(f"  [WARN] Source file does not exist at '{original_path}'. Skipping.")
                errors_encountered += 1
                continue
                
            destination_path = new_dir_path / filename
            try:
                shutil.move(original_path, destination_path)
                log.append(f"  -> Moved '{filename}' to '{new_dir_path}'")
                files_moved += 1
            except Exception as e:
                log.append(f"  [ERROR] Failed to move '{filename}'. Error: {e}")
                errors_encountered += 1
                
    total_files_in_plan = sum(len(f) for f in plan.values())
    summary = f"Execution complete. Moved {files_moved}/{total_files_in_plan} files. Encountered {errors_encountered} errors."

    return {"message": summary, "log": log}