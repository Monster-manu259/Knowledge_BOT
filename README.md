Q&A Assistance

A Streamlit-based app for RAG Flow.

Usage flow:
- Upload PDFs (sidebar) → process once → ask in chat window → receive answers from the LLM.
- Existing Pinecone index data is reused on next app runs, so re-upload is not required every time.

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

