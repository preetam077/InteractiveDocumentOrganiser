# app.py

import os
from flask import Flask, render_template, request, jsonify, session # 1. IMPORT 'session'
from dotenv import load_dotenv
from flask_session import Session  # 2. IMPORT 'Session' from flask_session

# Import your refactored functions
from summarygenerator import run_summary_generation
import fileorganizer

# Load environment variables
load_dotenv()

app = Flask(__name__)


# 4. ADD CONFIGURATION for server-side sessions
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24)) # Use env var or a random key
app.config["SESSION_TYPE"] = "sqlalchemy"
app.config["SESSION_PERMANENT"] = True # Make sessions persistent
app.config["SESSION_USE_SIGNER"] = True # Sign the session cookie
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sessions.db" # Database file name
app.config["SESSION_SQLALCHEMY_TABLE"] = "sessions" # Table name in the database

# 5. INITIALIZE the Session extension
Session(app)


@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

# --- Route Modifications ---
# The logic inside the routes now uses 'session' instead of 'app_state'

@app.route('/run_summary', methods=['POST'])
def run_summary():
    """Endpoint to trigger the summary generation."""
    data = request.get_json()
    base_path = data.get('base_path')
    
    try:
        max_workers = int(data.get('max_workers', 4))
    except (ValueError, TypeError):
        max_workers = 4

    if not base_path:
        return jsonify({"error": "BASE_PATH is required."}), 400
        
    results = run_summary_generation(base_path, max_workers=max_workers)
    return jsonify(results)

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    """Endpoint to get the initial file structure analysis."""
    if not os.path.exists('llm_input.json'):
        return jsonify({"error": "llm_input.json not found. Please run the summary generator first."}), 400
        
    results = fileorganizer.get_initial_analysis()
    
    if 'error' not in results:
        # CHANGE: Store data in the user's session
        session['all_docs'] = results.get('all_docs')
        session['current_analysis'] = results.get('analysis')
        
    return jsonify({"analysis": results.get('analysis')})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Endpoint for the user to ask a follow-up question."""
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided."}), 400
    
    # CHANGE: Retrieve context from the session
    all_docs = session.get('all_docs')
    current_analysis = session.get('current_analysis')
    
    if not all_docs or not current_analysis:
        return jsonify({"error": "Analysis context not found. Please run the analysis first."}), 400

    results = fileorganizer.answer_a_question(question, all_docs, current_analysis)
    return jsonify(results)
    
@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    """Endpoint to get the organization plan."""
    # CHANGE: Retrieve context from the session
    all_docs = session.get('all_docs')
    current_analysis = session.get('current_analysis')

    if not all_docs or not current_analysis:
        return jsonify({"error": "Analysis not performed yet. Please get analysis first."}), 400
    
    results = fileorganizer.get_organization_plan(all_docs, current_analysis)
    
    if 'error' not in results:
        # CHANGE: Save the plan to the session
        session['plan'] = results.get('plan')
        
    return jsonify(results)

@app.route('/execute_plan', methods=['POST'])
def execute_plan():
    """Endpoint to execute the file moving."""
    data = request.get_json()
    destination_root = data.get('destination_root')

    # CHANGE: Retrieve plan and docs from the session
    plan = session.get('plan')
    all_docs = session.get('all_docs')

    if not destination_root:
        return jsonify({"error": "DESTINATION_ROOT is required."}), 400
    if not plan:
        return jsonify({"error": "Organization plan not generated yet."}), 400
        
    results = fileorganizer.execute_the_plan(plan, all_docs, destination_root)

    # After a successful execution with no errors, delete the summary file.
    if 'error' not in results and results.get('errors_encountered') == 0:
        if os.path.exists('llm_input.json'):
            try:
                os.remove('llm_input.json')
                print("Successfully removed llm_input.json after successful plan execution.")
            except OSError as e:
                # Log the error, but don't fail the request. The user's files were moved.
                print(f"Error removing llm_input.json after execution: {e}")

    return jsonify(results)

@app.route('/reset', methods=['POST'])
def reset_state():
    """Clears server-side state for the current user."""
    try:
        # CHANGE: Clear the session data for this user
        session.pop('chat_history', None)
        session.clear()
        
        # Deleting the intermediate file can remain, as it's not user-specific
        if os.path.exists('llm_input.json'):
            os.remove('llm_input.json')
            
        print("User session cleared and llm_input.json removed.")
        return jsonify({"status": "success", "message": "Your session has been reset."})
    except Exception as e:
        print(f"Error during reset: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)