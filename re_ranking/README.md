# Re-Ranking Strategy for RAG

## Overview
This module implements an intelligent document re-ranking system for Retrieval-Augmented Generation (RAG). It uses LLMs and cross-encoders to re-rank retrieved documents by relevance, improving the quality of context passed to the language model.

## Features
- **PDF Document Encoding**: Loads PDFs and creates vector embeddings using OpenAI embeddings
- **Semantic Search**: Uses FAISS vector database for initial document retrieval
- **LLM-Based Re-ranking**: Leverages GPT-4o to score document relevance to queries
- **Custom Retriever**: Implements a custom LangChain retriever with intelligent ranking
- **Cross-Encoder Support**: Infrastructure for cross-encoder based re-ranking
- **Configurable Chunking**: Adjustable chunk size and overlap for document splitting

## Features Detail

### Document Encoding
Converts PDF documents into vector embeddings and stores them in a FAISS index for efficient retrieval.

### Re-ranking Pipeline
1. **Initial Retrieval**: Retrieves top-k documents using semantic similarity
2. **LLM Scoring**: Uses GPT-4o to score relevance on a scale of 1-10
3. **Re-ranking**: Sorts documents by relevance scores
4. **Top-N Selection**: Returns the most relevant documents

### Custom Retriever
Extends LangChain's BaseRetriever to implement the re-ranking logic with FAISS vector store integration.

## Usage

### Basic Setup
1. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Load and rank documents:
```python
from re_ranking import CustomRetriever, encode_pdf

# Encode your PDF
vectorstore = encode_pdf("path/to/your/document.pdf")

# Create retriever
retriever = CustomRetriever(vectorstore=vectorstore)

# Retrieve and rank documents
query = "Your question here"
relevant_docs = retriever._get_relevant_documents(query, num_docs=3)
```

## Configuration
- **LLM Model**: GPT-4o
- **Temperature**: 0 (deterministic)
- **Max Tokens**: 4000
- **Default Chunk Size**: 1000
- **Default Chunk Overlap**: 200
- **Initial Retrieval**: Top 30 documents
- **Final Output**: Top 3 documents (configurable)

## Main Components

### `encode_pdf(path: str, chunk_size: int = 1000, chunk_overlap: int = 200)`
Loads a PDF and creates a FAISS vector store.

**Parameters:**
- `path` (str): Path to the PDF file
- `chunk_size` (int): Size of text chunks
- `chunk_overlap` (int): Overlap between chunks

**Returns:**
- FAISS vectorstore with encoded documents

### `rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]`
Re-ranks documents based on LLM relevance scoring.

**Parameters:**
- `query` (str): The user query
- `docs` (List[Document]): Documents to re-rank
- `top_n` (int): Number of top documents to return

**Returns:**
- List of re-ranked documents sorted by relevance

### `CustomRetriever`
LangChain-compatible retriever that performs semantic search followed by LLM-based re-ranking.

## Dependencies
See `requirements.txt` for complete dependency list.

## Architecture
```
User Query
    ↓
FAISS Semantic Search (top-30)
    ↓
LLM Relevance Scoring
    ↓
Re-ranking & Sorting
    ↓
Top-N Results
    ↓
LLM Answer Generation
```

## Performance Considerations
- Initial retrieval uses fast vector similarity search
- LLM-based re-ranking is more expensive but improves quality
- Consider adjusting `top_n` parameter based on latency requirements
- Caching embeddings improves repeated queries on the same documents

## Use Cases
- High-quality document ranking for QA systems
- Filtering out false positives from semantic search
- Improving answer quality in RAG pipelines
- Multi-stage retrieval with quality control
