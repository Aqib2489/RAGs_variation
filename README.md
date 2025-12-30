# RAG Strategies Collection

A collection of advanced Retrieval-Augmented Generation (RAG) strategies and techniques for improving document retrieval and response quality in LLM-based systems.

## 📁 Strategies Overview

### 1. Chunking Strategy (`/chunking`)
Evaluates optimal document chunking sizes for RAG systems by measuring response quality metrics.

**Key Features:**
- Automated evaluation dataset generation
- Faithfulness and relevancy scoring using GPT-4o
- Response time benchmarking
- Chunk size optimization

**Tech Stack:** LlamaIndex, OpenAI GPT-4o, GPT-3.5-Turbo

**Use Case:** Determine the ideal chunk size for your specific documents to balance context quality and retrieval speed.

---

### 2. Query Transformation (`/query_change`)
Implements intelligent query reformulation techniques to improve retrieval accuracy.

**Key Features:**
- **Query Rewriting**: Reformulates queries to be more specific and detailed
- **Step-Back Prompting**: Generates broader queries for background context
- **Hybrid Retrieval**: Combines FAISS semantic search with BM25 keyword search

**Tech Stack:** LangChain, OpenAI GPT-4o, FAISS, BM25

**Use Case:** Enhance retrieval quality by transforming ambiguous or under-specified user queries into more effective search queries.

---

### 3. Re-Ranking Strategy (`/re_ranking`)
Implements LLM-based document re-ranking to improve relevance of retrieved context.

**Key Features:**
- Semantic search with FAISS vector database
- GPT-4o powered relevance scoring (1-10 scale)
- Custom LangChain retriever with intelligent ranking
- Two-stage retrieval pipeline

**Tech Stack:** LangChain, OpenAI GPT-4o, FAISS, Sentence Transformers

**Use Case:** Filter and prioritize the most relevant documents from initial retrieval results to improve answer quality.

---

## 🚀 Quick Start

Each folder contains its own README with detailed documentation. To get started:

1. **Navigate to a strategy folder:**
   ```bash
   cd chunking  # or query_change, or re_ranking
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Run the code:**
   ```bash
   python chunking.py  # or the respective Python file
   ```

## 📊 Strategy Comparison

| Strategy | Purpose | Complexity | Impact |
|----------|---------|------------|--------|
| **Chunking** | Optimize document splitting | Low | High - Foundation for retrieval |
| **Query Transform** | Improve query effectiveness | Medium | High - Better retrieval coverage |
| **Re-Ranking** | Prioritize relevant results | Medium | Very High - Quality filtering |

## 🎯 When to Use Each Strategy

- **Chunking**: Start here - essential for any RAG system
- **Query Transformation**: Use when queries are ambiguous or need context expansion
- **Re-Ranking**: Apply when initial retrieval returns too many false positives

## 🔗 Combining Strategies

These strategies can be combined for optimal results:
1. **Chunking** → Properly split your documents
2. **Query Transform** → Reformulate user queries
3. **Re-Ranking** → Filter and rank retrieved chunks

## 📝 Requirements

- Python 3.8+
- OpenAI API key
- PDF documents for testing (update paths in each script)

## 🤝 Contributing

Feel free to fork and experiment with different configurations and improvements!

## 📄 License

Open source - feel free to use and modify for your projects.
