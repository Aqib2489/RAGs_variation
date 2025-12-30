# Install required packages
# !pip install langchain langchain-openai langchain-community python-dotenv pypdf faiss-cpu rank-bm25

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains import create_retrieval_chain
from rank_bm25 import BM25Okapi
from typing import List, Dict
import numpy as np

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

re_write_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# Create a prompt template for query rewriting
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

# Create an LLMChain for query rewriting
query_rewriter = query_rewrite_prompt | re_write_llm

def rewrite_query(original_query):
    """
    Rewrite the original query to improve retrieval.
    
    Args:
    original_query (str): The original user query
    
    Returns:
    str: The rewritten query
    """
    response = query_rewriter.invoke(original_query)
    return response.content


# example query over the understanding climate change dataset
original_query = "What are the impacts of climate change on the environment?"
# rewritten_query = rewrite_query(original_query)
# print("Original query:", original_query)
# print("\nRewritten query:", rewritten_query)


step_back_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)


# Create a prompt template for step-back prompting
step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

Original query: {original_query}

Step-back query:"""

step_back_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=step_back_template
)

# Create an LLMChain for step-back prompting
step_back_chain = step_back_prompt | step_back_llm

def generate_step_back_query(original_query):
    """
    Generate a step-back query to retrieve broader context.
    
    Args:
    original_query (str): The original user query
    
    Returns:
    str: The step-back query
    """
    response = step_back_chain.invoke(original_query)
    return response.content


# example query over the understanding climate change dataset
original_query = "What are the impacts of climate change on the environment?"
# step_back_query = generate_step_back_query(original_query)
# print("Original query:", original_query)
# print("\nStep-back query:", step_back_query)


sub_query_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# Create a prompt template for sub-query decomposition
subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Original query: {original_query}

example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?"""


subquery_decomposition_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=subquery_decomposition_template
)

# Create an LLMChain for sub-query decomposition
subquery_decomposer_chain = subquery_decomposition_prompt | sub_query_llm

def decompose_query(original_query: str):
    """
    Decompose the original query into simpler sub-queries.
    
    Args:
    original_query (str): The original complex query
    
    Returns:
    List[str]: A list of simpler sub-queries
    """
    response = subquery_decomposer_chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    return sub_queries


# example query over the understanding climate change dataset
# original_query = "What are the impacts of climate change on the environment?"
# sub_queries = decompose_query(original_query)
# print("\nSub-queries:")
# for i, sub_query in enumerate(sub_queries, 1):
#     print(sub_query)


# ========================================================================================
# HYBRID RETRIEVER WITH RECIPROCAL RANK FUSION (RRF)
# ========================================================================================

class HybridRetrieverRRF:
    """
    Hybrid Retriever that combines BM25 (keyword-based) and Vector Search (semantic)
    using Reciprocal Rank Fusion (RRF) for better retrieval results.
    """
    
    def __init__(self, documents, embeddings, k=60):
        """
        Initialize the Hybrid Retriever with RRF.
        
        Args:
            documents: List of Document objects
            embeddings: Embedding model for vector search
            k: Constant for RRF formula (default: 60)
        """
        self.documents = documents
        self.embeddings = embeddings
        self.k = k
        
        # Create BM25 retriever (keyword-based)
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 10  # Retrieve top 10 documents
        
        # Create FAISS vector store retriever (semantic search)
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
    
    def reciprocal_rank_fusion(self, retrieval_results_list: List[List], k=60) -> List:
        """
        Apply Reciprocal Rank Fusion to combine multiple retrieval results.
        
        RRF Formula: RRF_score(d) = Σ (1 / (k + rank(d)))
        where rank(d) is the rank of document d in each retrieval result.
        
        Args:
            retrieval_results_list: List of retrieval results from different retrievers
            k: Constant for RRF formula (default: 60)
            
        Returns:
            List of documents sorted by RRF score
        """
        # Dictionary to store RRF scores for each document
        rrf_scores = {}
        
        # Calculate RRF scores
        for results in retrieval_results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.page_content  # Use content as identifier
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {"score": 0, "doc": doc}
                rrf_scores[doc_id]["score"] += 1 / (k + rank)
        
        # Sort documents by RRF score (descending)
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # Return the documents
        return [item["doc"] for item in sorted_docs]
    
    def retrieve(self, query: str, top_k: int = 5) -> List:
        """
        Retrieve documents using hybrid approach with RRF.
        
        Args:
            query: Search query
            top_k: Number of top documents to return
            
        Returns:
            List of top-k documents after RRF
        """
        # Get results from BM25 retriever
        bm25_results = self.bm25_retriever.invoke(query)
        
        # Get results from vector retriever
        vector_results = self.vector_retriever.invoke(query)
        
        # Apply RRF to combine results
        fused_results = self.reciprocal_rank_fusion(
            [bm25_results, vector_results], 
            k=self.k
        )
        
        # Return top-k results
        return fused_results[:top_k]


def load_and_process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load and process a PDF document into chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Loaded {len(documents)} pages from PDF")
    print(f"Split into {len(chunks)} chunks")
    
    return chunks


def build_rag_system_with_query_transformation(pdf_path: str, query: str, transformation_type: str = "rewrite"):
    """
    Build a complete RAG system with query transformation and hybrid retrieval using RRF.
    
    Args:
        pdf_path: Path to the PDF file
        query: Original user query
        transformation_type: Type of query transformation ('rewrite', 'step_back', or 'decompose')
        
    Returns:
        Final answer from the RAG system
    """
    print("=" * 80)
    print("RAG SYSTEM WITH QUERY TRANSFORMATION AND HYBRID RETRIEVAL (RRF)")
    print("=" * 80)
    
    # Step 1: Load and process PDF
    print("\n[1] Loading and processing PDF...")
    chunks = load_and_process_pdf(pdf_path)
    
    # Step 2: Initialize embeddings
    print("\n[2] Initializing embeddings...")
    embeddings = OpenAIEmbeddings()
    
    # Step 3: Create hybrid retriever with RRF
    print("\n[3] Creating hybrid retriever with Reciprocal Rank Fusion...")
    hybrid_retriever = HybridRetrieverRRF(chunks, embeddings, k=60)
    
    # Step 4: Apply query transformation
    print(f"\n[4] Applying query transformation (type: {transformation_type})...")
    print(f"Original Query: {query}")
    
    if transformation_type == "rewrite":
        transformed_query = rewrite_query(query)
        print(f"Rewritten Query: {transformed_query}")
        queries_to_search = [transformed_query]
        
    elif transformation_type == "step_back":
        step_back_q = generate_step_back_query(query)
        print(f"Step-back Query: {step_back_q}")
        queries_to_search = [query, step_back_q]  # Search both original and step-back
        
    elif transformation_type == "decompose":
        sub_queries = decompose_query(query)
        print("Sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"  {i}. {sq}")
        queries_to_search = sub_queries
        
    else:
        queries_to_search = [query]
    
    # Step 5: Retrieve relevant documents using hybrid retrieval
    print(f"\n[5] Retrieving relevant documents using hybrid retrieval (RRF)...")
    all_retrieved_docs = []
    for q in queries_to_search:
        docs = hybrid_retriever.retrieve(q, top_k=3)
        all_retrieved_docs.extend(docs)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc in all_retrieved_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    
    print(f"Retrieved {len(unique_docs)} unique document chunks")
    
    # Step 6: Generate answer using LLM
    print("\n[6] Generating answer using LLM...")
    context = "\n\n".join([doc.page_content for doc in unique_docs[:5]])  # Use top 5 chunks
    
    qa_template = """You are a helpful AI assistant. Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=1000)
    qa_chain = qa_prompt | llm
    
    response = qa_chain.invoke({"context": context, "question": query})
    answer = response.content
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    
    return {
        "original_query": query,
        "transformed_queries": queries_to_search,
        "retrieved_docs": unique_docs,
        "answer": answer
    }


# ========================================================================================
# INTERACTIVE LOOP FOR CONTINUOUS QUERYING
# ========================================================================================

def interactive_rag_loop():
    """
    Interactive loop that allows continuous querying of the RAG system.
    Load PDF once and query multiple times with different transformations.
    """
    print("=" * 80)
    print("INTERACTIVE RAG SYSTEM WITH HYBRID RETRIEVAL (RRF)")
    print("=" * 80)
    
    # Load PDF once
    pdf_path = "C:\\Users\\aqib8\\Desktop\\CV\\AI\\RAG_Repo\\Project.pdf"
    
    print(f"\nLoading PDF: {pdf_path}")
    chunks = load_and_process_pdf(pdf_path)
    
    print("\nInitializing embeddings and hybrid retriever...")
    embeddings = OpenAIEmbeddings()
    hybrid_retriever = HybridRetrieverRRF(chunks, embeddings, k=60)
    
    print("\n" + "=" * 80)
    print("PDF loaded! You can now ask questions.")
    print("=" * 80)
    
    # Interactive loop
    while True:
        print("\n" + "-" * 80)
        print("\nOptions:")
        print("1. Ask a question (with query transformation)")
        print("2. Change PDF")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "3":
            print("\nExiting RAG system. Goodbye!")
            break
        
        elif choice == "2":
            pdf_path = input("\nEnter new PDF path: ").strip()
            try:
                print(f"\nLoading PDF: {pdf_path}")
                chunks = load_and_process_pdf(pdf_path)
                hybrid_retriever = HybridRetrieverRRF(chunks, embeddings, k=60)
                print("PDF loaded successfully!")
            except Exception as e:
                print(f"Error loading PDF: {e}")
                print("Continuing with previous PDF...")
            continue
        
        elif choice == "1":
            # Get query from user
            print("\n" + "=" * 80)
            query = input("Enter your question: ").strip()
            
            if not query:
                print("Empty query. Please try again.")
                continue
            
            # Choose transformation type
            print("\nQuery Transformation Options:")
            print("1. Rewrite (reformulate query)")
            print("2. Step-back (add broader context query)")
            print("3. Decompose (break into sub-queries)")
            print("4. None (use original query)")
            
            trans_choice = input("\nChoose transformation (1-4, default=1): ").strip() or "1"
            
            transformation_map = {
                "1": "rewrite",
                "2": "step_back",
                "3": "decompose",
                "4": "none"
            }
            transformation_type = transformation_map.get(trans_choice, "rewrite")
            
            # Apply query transformation
            print(f"\nOriginal Query: {query}")
            
            if transformation_type == "rewrite":
                transformed_query = rewrite_query(query)
                print(f"Rewritten Query: {transformed_query}")
                queries_to_search = [transformed_query]
                
            elif transformation_type == "step_back":
                step_back_q = generate_step_back_query(query)
                print(f"Step-back Query: {step_back_q}")
                queries_to_search = [query, step_back_q]
                
            elif transformation_type == "decompose":
                sub_queries = decompose_query(query)
                print("Sub-queries:")
                for i, sq in enumerate(sub_queries, 1):
                    print(f"  {i}. {sq}")
                queries_to_search = sub_queries
            else:
                queries_to_search = [query]
            
            # Retrieve documents
            print("\nRetrieving relevant documents...")
            all_retrieved_docs = []
            for q in queries_to_search:
                docs = hybrid_retriever.retrieve(q, top_k=3)
                all_retrieved_docs.extend(docs)
            
            # Remove duplicates
            seen = set()
            unique_docs = []
            for doc in all_retrieved_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
            
            print(f"Retrieved {len(unique_docs)} unique document chunks")
            
            # Generate answer
            print("\nGenerating answer...")
            context = "\n\n".join([doc.page_content for doc in unique_docs[:5]])
            
            qa_template = """You are a helpful AI assistant. Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
            
            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=qa_template
            )
            
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=1000)
            qa_chain = qa_prompt | llm
            
            response = qa_chain.invoke({"context": context, "question": query})
            answer = response.content
            
            print("\n" + "=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# ========================================================================================
# EXAMPLE USAGE
# ========================================================================================

if __name__ == "__main__":
    # Run interactive loop
    interactive_rag_loop()

