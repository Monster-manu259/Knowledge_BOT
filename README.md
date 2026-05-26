Q&A Assistance

A Streamlit-based app for RAG Flow.

Usage flow:
- Upload PDFs → extract text → split into chunks → embed & index → ask a question → receive an answer from the LLM.

Required environment variables:
- `GROQ_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`

Setup:
- Create and activate a virtual environment:
	- `python -m venv venv`
	- `venv\\Scripts\\activate` (Windows PowerShell)
- Install dependencies:
	- `pip install -r requirements.txt`

Run (PowerShell):
- Set environment variables (examples):
	- `$env:GROQ_API_KEY="your_groq_key"`
	- `$env:PINECONE_API_KEY="your_pinecone_key"`
	- `$env:PINECONE_INDEX_NAME="your_index_name"`
- Start the Streamlit app:
	- `streamlit run app.py`

