# Chunking Strategy for RAG

## Overview
This module evaluates different document chunking strategies for Retrieval-Augmented Generation (RAG) systems. It tests how various chunk sizes affect response quality, faithfulness, and relevancy when using LLMs to answer questions about document content.

## Features
- **PDF Document Loading**: Loads PDF documents using LlamaIndex
- **Evaluation Dataset Generation**: Automatically generates evaluation questions from document content
- **Response Quality Metrics**: 
  - Response time measurement
  - Faithfulness evaluation (whether responses are supported by source context)
  - Relevancy evaluation (whether responses are relevant to queries)
- **Chunk Size Optimization**: Evaluates multiple chunk sizes to find optimal performance
- **GPT-4 Evaluation**: Uses GPT-4o for high-quality evaluation metrics
- **GPT-3.5-Turbo Inference**: Uses faster, cheaper model for generating responses

## Usage

### Basic Setup
1. Ensure you have a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

2. Update the `pdf_path` variable to point to your document:
```python
pdf_path = r"C:\Users\aqib8\Desktop\CV\AI\RAG_Repo\Project.pdf"
```

3. Run the script:
```bash
python chunking.py
```

### Key Functions

#### `evaluate_response_time_and_accuracy(chunk_size, eval_questions)`
Evaluates performance metrics for a specific chunk size.

**Parameters:**
- `chunk_size` (int): The size of text chunks to split documents into
- `eval_questions` (list): List of evaluation questions to test

**Returns:**
- Tuple containing (average_response_time, average_faithfulness, average_relevancy)

## Configuration
- **Number of Evaluation Questions**: `num_eval_questions = 10`
- **LLM Models**: 
  - Evaluation: GPT-4o
  - Response Generation: GPT-3.5-Turbo
  - Evaluation Temperature: 0 (deterministic)

## Dependencies
See `requirements.txt` for complete dependency list.

## Output
The script generates evaluation metrics that help determine optimal chunk sizes for your specific documents and use case.
