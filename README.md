PDF RAG System
===============

A Python-based Retrieval-Augmented Generation (RAG) system that extracts text from PDF files,
splits it into chunks, stores it in a vector database (Chroma), and answers questions using a
language model.

Features
--------

- Extract text from multiple PDF files
- Clean and split text into smaller chunks with metadata
- Store chunks in ChromaDB with embeddings
- Use sentence-transformers/all-MiniLM-L6-v2 for embeddings
- Answer questions using google/flan-t5-base text-to-text model
- Returns answer along with reference chunks and metadata (file name, page number)

Requirements
------------

- Python 3.10+
- Libraries (can be installed via requirements.txt):
  - PyPDF2
  - transformers
  - torch
  - chromadb
  - sentence-transformers
  - langchain

Install all dependencies:

pip install -r requirements.txt



Setup
-----

1. Clone the repository:

   git clone https://github.com/danajayyad/pdf-rag-system.git
   cd pdf-rag-system

2. Make sure your PDF files are in the project folder or provide the full path.

3. Ensure requirements.txt is installed.

Usage
-----

Run the main script:

python main.py


You will be prompted to:

1. Enter PDF file names (comma separated), for example:

   example.pdf, project_report.pdf

2. Enter your question, for example:

  What are the key milestones of the AI project?


Example Output
--------------

-----RESULTS-----
Q: What are the key milestones of the AI project?
A: The AI project consists of three main milestones: 
1. Data Collection and Cleaning – gathering and preprocessing datasets.
2. Model Development – designing, training, and validating AI models.
3. Deployment and Monitoring – deploying models to production and tracking performance metrics.

-----REFERENCES-----
File: project_report.pdf Page: 3
Text: Milestone 1: Data Collection and Cleaning – We gathered data from multiple sources...

File: project_report.pdf Page: 5
Text: Milestone 2: Model Development – The AI team trained several models including CNNs...

File: project_report.pdf Page: 7
Text: Milestone 3: Deployment and Monitoring – Models were deployed in a cloud environment...




Notes
-----

- The system uses ChromaDB for vector storage and retrieval.
- Make sure you have enough GPU memory if using large models.
- langchain is used only for text splitting; you can replace it with other splitters if needed.
