# Medical-QA-Assistant-with-Fine-Tuned-Mistral-RAG

A powerful Medical Question Answering Assistant leveraging Retrieval-Augmented Generation (RAG) and a fine-tuned Mistral language model. This project is designed to provide accurate, context-aware answers to medical queries using advanced natural language processing and retrieval techniques.

## Features
- Retrieval-Augmented Generation (RAG) for enhanced answer accuracy
- Fine-tuned Mistral LLM for medical domain
- Modular codebase for easy extension
- PubMed integration for up-to-date medical information
- Docker support for easy deployment

## Project Structure
```
app.py                # Main application entry point
llm/                  # Language model and generator modules
rag/                  # Indexing and retrieval modules
utils/                # Utility scripts (e.g., PubMed integration)
requirements.txt      # Python dependencies
Dockerfile            # Containerization support
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- (Optional) Docker

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/SachinMosambe/Medical-QA-Assistant-with-Fine-Tuned-Mistral-RAG.git
   cd Medical-QA-Assistant-with-Fine-Tuned-Mistral-RAG
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application
```
python app.py
```

Or with Docker:
```
docker build -t medical-qa-assistant .
docker run -p 8000:8000 medical-qa-assistant
```

## Usage
- Start the application and interact with the assistant via the provided interface (CLI, API, or web UI as implemented).
- Ask medical questions and receive contextually relevant answers.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.


