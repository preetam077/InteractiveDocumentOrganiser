# fileorganizer.py (Corrected Version)

import json
import os
import shutil
import google.generativeai as genai  # Use the standard import
from dotenv import load_dotenv
from pathlib import Path

# --- ONE-TIME SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- CHANGE #1: Switch from genai.Client to genai.configure ---
# This is the modern and correct way to initialize the library.
try:
    if not GOOGLE_API_KEY:
        raise ValueError("Error: GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"CRITICAL ERROR initializing Google GenAI: {e}")
    # We will handle the uninitialized state within each function.

# --- HELPER FUNCTIONS (unchanged) ---

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

# --- NEW TOOL FUNCTION (unchanged) ---

def search_documents(all_docs: list, query: str):
    """
    Searches document summaries for a specific query string.
    Only returns documents where the query is found in the summary.
    """
    print(f"\n---> TOOL EXECUTED: Searching for documents containing '{query}'...")
    if not all_docs:
        return {"error": "Document data is not available for search."}
    matches = []
    for doc in all_docs:
        if query.lower() in doc.get("summary", "").lower():
            matches.append({"file_path": doc.get("file_path"), "summary": doc.get("summary")})
    print(f"---> TOOL: Found {len(matches)} matching documents.")
    return {"found_documents": matches}


# --- MAIN CALLABLE FUNCTIONS (Updated) ---

def get_initial_analysis():
    """Loads data from llm_input.json and gets the initial AI analysis."""
    if not GOOGLE_API_KEY:
        return {"error": "Google GenAI is not initialized. Check API key."}
        
    all_docs = load_document_data()
    if not all_docs:
        return {"error": "Could not load llm_input.json. Run the summary generator first."}

    documents_str = "\n".join([
        f"- File Path: {doc['file_path']}\n  Type: {doc.get('type', 'N/A')}\n  Summary: {doc['summary']}\n"
        for doc in all_docs
    ])
    prompt = f"""
    You are an expert file organization assistant. Your task is to analyze the current file structure based on the provided file paths, types, and summaries, explain why the existing placement and structure may not be the best optimized, and convince the user that a new organization would be beneficial.

    **File Information:**
    {documents_str}

    **Instructions:**
    Respond ONLY with a concise analysis (200-400 words) that includes:
    - A description of the current structure (e.g., how files are grouped, any patterns in directories).
    - Why it may not be optimal (e.g., scattered files, lack of logical grouping by project/year/topic, redundancy, difficulty in navigation).
    - Suggestions for improvement (high-level, without providing the full plan yet).
    - A convincing argument on the benefits of reorganizing (e.g., easier access, better scalability, reduced search time).

    Do not provide a JSON plan, file tree, or any reorganization details yet. Focus on analysis and persuasion.
    """
    try:
        # --- CHANGE #2: Initialize model and call generate_content ---
        model = genai.GenerativeModel('')
        response = model.generate_content(prompt)
        analysis_text = response.text.strip()
        return {"analysis": analysis_text, "all_docs": all_docs}
    except Exception as e:
        return {"error": f"Failed to get analysis from AI: {e}"}

# Replace the existing answer_a_question function with this final, most compatible version.

def answer_a_question(question: str, all_docs: list, current_analysis: str):
    """Answers a user question using the scalable tool-use approach."""
    if not GOOGLE_API_KEY:
        return {"error": "Google GenAI is not initialized."}

    try:
        # Step 1: Initialize the model WITH the tool definition
        # The library inspects the function to create the schema.
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite-latest',
            tools=[search_documents]
        )
        
        # The generic prompt remains the same.
        prompt = f"""
        You are an expert file assistant. Your primary function is to answer questions about a collection of documents based on their summaries.

        **Your ONLY way to access the specific content of these document summaries is by using the `search_documents` tool.** You cannot see the files directly.

        **CRITICAL INSTRUCTION:** To answer any user question that requires looking for specific information inside the documents (such as names, topics, ingredients, skills, etc.), you MUST call the `search_documents` function with a relevant keyword.
        
        **Example of how to think:** If your analysis indicates the documents are recipes and the user asks 'what dishes use garlic?', a good query for the tool would be 'garlic'. If the analysis suggests they are resumes and the user asks for 'who is a manager?', a good query would be 'manager'.

        You have already performed a high-level analysis of all the documents. Use this analysis to understand the general context and topic of the document collection:
        **Your Previous Analysis:**
        "{current_analysis}"

        Now, answer the user's question based on these instructions.
        **User's Question:**
        "{question}"
        """

        # --- CHANGE: We now use model.generate_content directly instead of a chat session ---
        print("--- Sending question to model to decide on tool use...")
        response = model.generate_content(prompt, tools=[search_documents])
        
        response_part = response.candidates[0].content.parts[0]
        if not hasattr(response_part, 'function_call'):
            print("--- Model answered directly without tool.")
            return {"answer": response.text.strip()}

        function_call = response_part.function_call
        print(f"--- Model wants to call tool: {function_call.name}")

        if function_call.name == "search_documents":
            query = function_call.args.get("query")
            tool_result = search_documents(all_docs=all_docs, query=query)

            # --- CHANGE: Manually construct the conversation history for the second call ---
            # This is the most stable way to handle the conversation, avoiding helper classes.
            conversation_history = [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "model", "parts": [response_part]},
                {
                    "role": "function",
                    "parts": [
                        {
                            "function_response": {
                                "name": function_call.name,
                                "response": tool_result
                            }
                        }
                    ]
                }
            ]

            print("--- Sending tool result back to model for final answer...")
            final_response = model.generate_content(
                conversation_history,
                tools=[search_documents]
            )
            return {"answer": final_response.text.strip()}
        else:
            return {"error": f"Model tried to call an unknown function: {function_call.name}"}

    except Exception as e:
        print(f"An exception occurred in answer_a_question: {e}")
        return {"error": f"Failed to get an answer from the AI: {e}"}


def get_organization_plan(all_docs, current_analysis):
    """Gets the file organization plan, tree, and reasoning from the AI."""
    if not GOOGLE_API_KEY:
        return {"error": "Google GenAI is not initialized."}
        
    documents_str = "\n".join([
        f"- File: {os.path.basename(doc['file_path'])}\n  Summary: {doc['summary']}\n"
        for doc in all_docs
    ])
    prompt = f"""
    You are an expert file organization assistant. Based on the following analysis of the current file structure, your task is to organize the files listed below into a logical folder structure and provide a JSON plan, an ASCII file tree representation, and a reasoning section explaining the organization.

    **Previous Analysis of Current Structure:**
    {current_analysis}

    **File Information:**
    {documents_str}

    **Instructions:**
    Respond ONLY with a single output containing three sections, separated clearly. Do not include any additional text, explanations, or markdown formatting outside the specified structure. DO NOT CHANGE THE NAME OF FILES; KEEP THEM AS THEY ARE.

    1. **JSON Plan**:
       - A JSON object where each key is the proposed new directory path (e.g., "Case_Studies/2020_Grimmen_Vegetation").
       - Each value is a list of filenames (e.g., ["Case Study_Cutting Vegetation_2020.docx", "Fassade nach Cutting.png"]) to be moved into that directory.
       - Use forward slashes (/) for directory paths.

    2. **ASCII File Tree**:
       - After the JSON, include a line with exactly "-----" to separate sections.
       - Provide an ASCII file tree representation of the same structure.

    3. **Reasoning**:
       - After the file tree, include another line with exactly "-----" to separate sections.
       - Provide a concise explanation (100-200 words) of why this organization plan was chosen.

    **Output Format**:
    ```json
    {{
      "Case_Studies/2018_Cadolzburg": [
        "Case Study_Cadolzburg_v1.docx",
        "ZAE_Modulliste.pdf"
      ]
    }}
    -----
    Case_Studies/
    └── 2018_Cadolzburg/
        ├── Case Study_Cadolzburg_v1.docx
        └── ZAE_Modulliste.pdf
    -----
    The files are organized by project and year...
    ```

    Ensure all files from the input are included in both the JSON and the file tree.
    """
    try:
        # --- CHANGE #4: Update this function as well ---
        model = genai.GenerativeModel('gemini-2.5-flash-lite-latest')
        response = model.generate_content(prompt)
        response_text = response.text.strip().replace('```json', '').replace('```', '')
        parts = response_text.split('-----', 2)
        if len(parts) != 3:
            raise ValueError("Invalid response format from AI.")
        
        json_part, file_tree, reasoning = [part.strip() for part in parts]
        plan = json.loads(json_part)
        return {"plan": plan, "file_tree": file_tree, "reasoning": reasoning}
    except Exception as e:
        return {"error": f"Failed to get a valid plan from the AI: {e}"}


def execute_the_plan(plan, all_docs, destination_root_str):
    # This function does not call the AI, so it remains unchanged.
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
            log.append(f"[ERROR] Could not create directory '{new_dir_path}'. Skipping. Error: {e}")
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
