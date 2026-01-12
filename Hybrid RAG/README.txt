Hybrid RAG for MID Medicines Dataset (Fully Local)

This project implements a fully local Hybrid Retrieval-Augmented Generation (RAG) system over the MID Medicines Information Dataset (MID.xlsx).
It is designed to answer natural-language questions about medications using both structured SQL retrieval and semantic vector retrieval, followed by local LLM reasoning.

The system runs entirely offline and does not rely on any cloud APIs.


Technologies Used:

It combines:

Structured retrieval (SQLite / SQL)

Semantic retrieval (ChromaDB + vector embeddings)

Local LLM inference (Ollama – Mistral)

No cloud APIs. No paid services. Everything runs locally.


Features:

Loads MID.xlsx into SQLite

SQL query generation via a local LLM

Semantic search using bge-m3 embeddings

Vector storage with ChromaDB

Answer generation using Mistral (via Ollama)

Fully offline & privacy-preserving

Hybrid SQL + vector retrieval for better factual grounding


Installation:

Clone the repository:
	git clone https://github.com/MehrBSh/LLM-Projects.git
	cd Hybrid-RAG

Create a virtual environment (recommended):
	python -m venv venv
	source venv/bin/activate    # Linux / macOS
	venv\Scripts\activate       # Windows

Install dependencies
	pip install -r requirements.txt


Install and prepare Ollama:

Download Ollama from: https://ollama.com

Pull the required models:

	ollama pull mistral
	ollama pull bge-m3

Make sure Ollama is running:

	ollama serve


Dataset Setup:

Download MID.xlsx from the original source or dataset provider (https://data.mendeley.com/datasets/2vk5khfn6v/2)

Place it anywhere on your system

Update the path in the code:

	MID_PATH = "path/to/your/MID.xlsx"


⚠️ The dataset is not included in this repository.


Usage:

Run the main Hybrid RAG workflow:

	python hybridRAG.py

First Run Behavior:

On the first execution:

MID.xlsx is loaded into SQLite (mid.db)

Vector embeddings are generated

ChromaDB vector store is initialized

On subsequent runs, cached data is reused for faster startup.


Example Queries:

You can ask natural-language questions such as:

"What do you know about Aspirin?"


The answer will appear in the console.

Type exit to stop the program.


Hybrid RAG Workflow:

User provides a question

The system generates a safe SQL query using the local LLM

Relevant rows are retrieved from SQLite

The question is embedded using bge-m3

Semantic matches are retrieved from ChromaDB

Both SQL results and vector documents are passed to Mistral

The final answer is generated only from retrieved data


This hybrid approach improves:

Accuracy

Grounding

Resistance to hallucination


📁 File Structure
Hybrid-RAG/
├── hybridRAG.py          # Hybrid RAG pipeline
├── MID.xlsx              # Medicines dataset (not included)
├── mid.db                # SQLite database (auto-generated)
├── chroma_mid_db/        # Chroma vector store
├── README.md
└── requirements.txt


Safety & Guardrails:

SQL queries are restricted to SELECT only

No JOIN operations allowed

Only valid dataset columns can be used

Answers are generated strictly from retrieved data

No medical facts are hallucinated

