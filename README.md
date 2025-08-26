# Interactive Document Search & Organizer

This is a web-based tool that uses AI to help you make sense of and organize a messy folder of documents. It scans a local directory, generates intelligent summaries for various file types, and provides an interactive interface to search your documents with natural language and automatically organize them into a clean, new folder structure.

## Features

-   **Multi-Format Document Processing**: Scans and processes a wide range of file types including `.pdf`, `.docx`, `.pptx`, `.xlsx`, images (`.png`, `.jpg`), audio (`.mp3`), and more.
-   **AI-Powered Summaries**: Generates concise summaries for each document using sentence embeddings (`all-MiniLM-L6-v2`) to identify the most relevant sentences.
-   **Intelligent Analysis**: Leverages the Google Gemini Pro model to provide a high-level analysis of your current file structure, explaining why it's inefficient.
-   **Interactive Q&A**: Ask natural language questions about your documents (e.g., "Which resumes mention Python?") and get answers based on the generated summaries.
-   **Automated Organization Plan**: Generates a complete, logical folder structure plan based on the content of your files.
-   **Safe Execution**: Review the proposed plan (as a file tree and JSON) and reasoning before executing. The tool then physically moves the files to the new, organized destination folder.
-   **Web-Based UI**: A simple, step-by-step interface built with Flask that guides you through the entire process.

## Tech Stack

-   **Backend**: Python, Flask
-   **AI Model**: Google Gemini 1.5 Flash
-   **Embeddings**: `sentence-transformers` library
-   **Document Conversion**: `docling` library
-   **Frontend**: HTML, CSS, JavaScript (no frameworks)

## Project Structure

```
.
├── app.py                  # Main Flask application, handles routing and session management.
├── fileorganizer.py        # Contains all logic for interacting with the Gemini AI model.
├── summarygenerator.py     # Handles file scanning, text extraction, and summary generation.
├── requirements.txt        # List of Python dependencies.
├── .gitignore              # Specifies files for Git to ignore.
├── .env.example            # Example environment file template.
├── README.md               # This file.
└── templates/
    └── index.html          # The single-page frontend for the application.
```

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   A Google Gemini API Key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/preetam077/InteractiveDocumentOrganiser.git
    cd InteractiveDocumentOrganiser
    ```

2.  **Create and activate a virtual environment:**
    -   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    -   Copy the example file:
        ```bash
        copy .env.example .env
        ```
    -   Open the `.env` file in a text editor and add your `GOOGLE_API_KEY` and a new `SECRET_KEY`. For the secret key, you can generate a secure random key by running this in a Python terminal: `import os; print(os.urandom(24).hex())`

5.  **Run the application:**
    ```bash
    python app.py
    ```

6.  Open your web browser and navigate to `http://127.0.0.1:5000`.

## How to Use

The web interface is organized into a simple 4-step process:

1.  **Scan & Summarize**: Enter the full path to the directory you want to organize and click "Run Scan & Summary". Wait for the process to complete.
2.  **Analyze**: Click "Get AI Analysis". The AI will review the summaries and file paths and provide an overview of the current organizational state.
3.  **Generate Plan**:
    -   (Optional) Use the Q&A section to ask specific questions about your files.
    -   Click "Generate Organization Plan". The AI will return its proposed folder structure as a file tree and the reasoning behind its choices.
4.  **Execute**:
    -   Review the plan carefully.
    -   Enter a full path to a new, empty folder where the organized files will be placed.
    -   Click the **Execute Plan** button. A confirmation prompt will appear. Once confirmed, the tool will move the files from their original locations to the new, structured destination.
