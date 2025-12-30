# Query Transformation Strategies for RAG

## Overview
This module implements advanced query transformation techniques for Retrieval-Augmented Generation (RAG) systems. It provides methods to reformulate user queries to improve document retrieval accuracy and relevancy.

## Features
- **Query Rewriting**: Automatically reformulates queries to be more specific and detailed
- **Step-Back Prompting**: Generates broader, more general queries for retrieving background context
- **Multi-Retriever Support**: 
  - FAISS-based semantic vector search
  - BM25-based keyword/lexical search
  - Hybrid retrieval combining both approaches
- **PDF Document Processing**: Loads and splits PDF documents for retrieval
- **LangChain Integration**: Uses LangChain for robust LLM integration

## Features Detail

### Query Rewriting
Reformulates user queries to be more specific and likely to retrieve relevant information from the knowledge base.

**Example:**
```
Original: "What are the impacts of climate change?"
Rewritten: "What are the specific environmental, economic, and social impacts of climate change on different regions?"
```

### Step-Back Prompting
Generates broader queries that help retrieve background information and context.

**Example:**
```
Original: "What are the impacts of climate change on the environment?"
Step-Back: "What is climate change and how does it work?"
```

### Hybrid Retrieval
Combines semantic search (FAISS) with keyword-based search (BM25) for comprehensive document retrieval.

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

3. Use the query transformation functions:
```python
from query_transformation import rewrite_query, generate_step_back_query

original = "What are the impacts of climate change?"
rewritten = rewrite_query(original)
step_back = generate_step_back_query(original)
```

## Configuration
- **LLM Model**: GPT-4o
- **Temperature**: 0 (deterministic)
- **Max Tokens**: 4000

## Main Functions

### `rewrite_query(original_query: str) -> str`
Rewrites a query to improve retrieval effectiveness.

### `generate_step_back_query(original_query: str) -> str`
Generates a more general query for background context.

## Dependencies
See `requirements.txt` for complete dependency list.

## Use Cases
- Improving retrieval quality in RAG systems
- Multi-hop query answering
- Clarifying ambiguous user queries
- Retrieving broader context before specific answers
