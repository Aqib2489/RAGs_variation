import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field


# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Create data directory if needed
os.makedirs('data', exist_ok=True)


def encode_pdf(path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load and encode PDF into a vector store."""
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print(f"Loaded {len(documents)} pages and created {len(chunks)} chunks")
    return vectorstore


path = r"C:\Users\aqib8\Desktop\CV\AI\RAG_Repo\Project.pdf"

vectorstore = encode_pdf(path)

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")

def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)
    
    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))
    
    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]


# Create a custom retriever class
class CustomRetriever(BaseRetriever, BaseModel):
    
    vectorstore: Any = Field(description="Vector store for initial retrieval")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(query, initial_docs, top_n=num_docs)
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")


# Create the custom retriever
custom_retriever = CustomRetriever(vectorstore=vectorstore)

# Create an LLM for answering questions
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {input}

Answer:"""
)

# Create the retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(custom_retriever, document_chain)


# Cross encoder reranker implementation

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort documents by score
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top reranked documents
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")



# Create the cross-encoder retriever
cross_encoder_retriever = CrossEncoderRetriever(
    vectorstore=vectorstore,
    cross_encoder=cross_encoder,
    k=10,  # Retrieve 10 documents initially
    rerank_top_k=5  # Return top 5 after reranking
)

# Set up the LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {input}

Answer:"""
)

# Create the retrieval chain with the cross-encoder retriever
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(cross_encoder_retriever, document_chain)


# ========================================================================================
# INTERACTIVE QA LOOP
# ========================================================================================

def interactive_qa_loop():
    """
    Interactive loop for continuous question-answering on the PDF.
    Allows users to choose between LLM reranking and Cross-Encoder reranking.
    """
    print("\n" + "="*80)
    print("INTERACTIVE QA SYSTEM WITH RERANKING")
    print("="*80)
    print(f"PDF loaded: {path}")
    print(f"Total chunks: {len(vectorstore.docstore._dict)} documents indexed")
    
    # Create both retrievers
    custom_retriever = CustomRetriever(vectorstore=vectorstore)
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
{context}

Question: {input}

Answer:"""
    )
    
    # Create chains for both methods
    doc_chain_llm = create_stuff_documents_chain(llm, prompt)
    qa_chain_llm = create_retrieval_chain(custom_retriever, doc_chain_llm)
    
    doc_chain_cross = create_stuff_documents_chain(llm, prompt)
    qa_chain_cross = create_retrieval_chain(cross_encoder_retriever, doc_chain_cross)
    
    print("\n" + "="*80)
    print("Ready to answer questions!")
    print("="*80)
    
    while True:
        print("\n" + "-"*80)
        print("\nOptions:")
        print("1. Ask a question (LLM reranking)")
        print("2. Ask a question (Cross-Encoder reranking)")
        print("3. Compare both methods")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4":
            print("\nExiting QA system. Goodbye!")
            break
        
        elif choice in ["1", "2", "3"]:
            query = input("\nEnter your question: ").strip()
            
            if not query:
                print("Empty question. Please try again.")
                continue
            
            if choice == "1":
                # LLM reranking
                print("\n[Using LLM Reranking...]")
                result = qa_chain_llm.invoke({"input": query})
                
                print("\n" + "="*80)
                print("ANSWER (LLM Reranking):")
                print("="*80)
                print(result['answer'])
                print("\n" + "-"*80)
                print("Source Documents:")
                for i, doc in enumerate(result["context"][:3]):
                    print(f"\nDocument {i+1}:")
                    print(doc.page_content[:200] + "...")
                
            elif choice == "2":
                # Cross-Encoder reranking
                print("\n[Using Cross-Encoder Reranking...]")
                result = qa_chain_cross.invoke({"input": query})
                
                print("\n" + "="*80)
                print("ANSWER (Cross-Encoder Reranking):")
                print("="*80)
                print(result['answer'])
                print("\n" + "-"*80)
                print("Source Documents:")
                for i, doc in enumerate(result["context"][:3]):
                    print(f"\nDocument {i+1}:")
                    print(doc.page_content[:200] + "...")
                
            elif choice == "3":
                # Compare both methods
                print("\n[Comparing both reranking methods...]")
                
                result_llm = qa_chain_llm.invoke({"input": query})
                result_cross = qa_chain_cross.invoke({"input": query})
                
                print("\n" + "="*80)
                print("LLM RERANKING ANSWER:")
                print("="*80)
                print(result_llm['answer'])
                
                print("\n" + "="*80)
                print("CROSS-ENCODER RERANKING ANSWER:")
                print("="*80)
                print(result_cross['answer'])
                
                print("\n" + "-"*80)
                print("Retrieved documents comparison:")
                print(f"LLM method retrieved: {len(result_llm['context'])} docs")
                print(f"Cross-Encoder method retrieved: {len(result_cross['context'])} docs")
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


# ========================================================================================
# RUN THE INTERACTIVE LOOP
# ========================================================================================

if __name__ == "__main__":
    interactive_qa_loop()
