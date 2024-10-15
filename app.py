import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from ragatouille import RAGPretrainedModel
from typing import List, Tuple

# Model initialization
@st.cache_resource
def load_model():
    READER_MODEL_NAME = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=1000,
    )
    return READER_LLM, tokenizer

READER_LLM, tokenizer = load_model()

# Initialize reranker
@st.cache_resource
def load_reranker():
    return RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

RERANKER = load_reranker()

# Load and process documents
@st.cache_resource
def load_and_process_documents():
    loader = DirectoryLoader("path_to_your_documents", glob="**/*.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_processed = text_splitter.split_documents(documents)
    return docs_processed

docs_processed = load_and_process_documents()

# Create knowledge base
@st.cache_resource
def create_knowledge_base():
    embedding_model = HuggingFaceEmbeddings()
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy="COSINE"
    )
    return KNOWLEDGE_VECTOR_DATABASE, embedding_model

KNOWLEDGE_VECTOR_DATABASE, embedding_model = create_knowledge_base()

# Updated RAG prompt template
prompt_in_chat_format = [
    {
        "role": "system",
        "content": """You are an AI assistant specializing in analyzing PDF documents. Using the information from the provided PDF context, give a comprehensive answer to the question.
        Respond only to the question asked, ensuring your response is concise and relevant.
        If possible, reference specific page numbers or sections from the PDF.
        If the answer cannot be found in the PDF context, state that the information is not available in the provided documents.""",
    },
    {
        "role": "user",
        "content": """PDF Context:
        {context}
        ---
        Question: {question}""",
    },
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)

def answer_with_rag(
    question: str,
    llm: pipeline,
    knowledge_index: FAISS,
    reranker: RAGPretrainedModel,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[str]]:
    # Gather documents with retriever
    st.write("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text
    
    # Rerank results
    st.write("=> Reranking documents...")
    relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
    relevant_docs = [doc["content"] for doc in relevant_docs]
    
    # Build the final prompt
    context = "\nExtracted PDF content:\n"
    context += "".join([f"Section {str(i+1)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
    
    # Generate an answer
    st.write("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]
    
    return answer, relevant_docs

def visualize_embeddings(user_query):
    query_vector = embedding_model.embed_query(user_query)
    
    # Create 2D embeddings for visualization
    reducer = UMAP(n_components=2, random_state=42)
    
    embeddings_2d = [
        list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0])
        for idx in range(len(docs_processed))
    ] + [query_vector]
    
    # Project the embeddings to 2D space
    documents_projected = reducer.fit_transform(np.array(embeddings_2d))
    
    # Create a DataFrame for visualization
    df = pd.DataFrame.from_dict(
        [
            {
                "x": documents_projected[i, 0],
                "y": documents_projected[i, 1],
                "source": docs_processed[i].metadata["source"].split("/")[-1],
                "extract": docs_processed[i].page_content[:100] + "...",
                "symbol": "circle",
                "size_col": 4,
            }
            for i in range(len(docs_processed))
        ]
        + [
            {
                "x": documents_projected[-1, 0],
                "y": documents_projected[-1, 1],
                "source": "User query",
                "extract": user_query,
                "size_col": 100,
                "symbol": "star",
            }
        ]
    )
    
    # Create the scatter plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        symbol="symbol",
        size="size_col",
        hover_data=["extract"],
        title="Document Embeddings Visualization",
    )
    
    return fig

# Streamlit UI
st.title("RAG Question Answering System")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        answer, relevant_docs = answer_with_rag(question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER)
        
        st.subheader("Answer:")
        st.write(answer)
        
        st.subheader("Relevant Documents:")
        for i, doc in enumerate(relevant_docs, 1):
            st.write(f"Document {i}:")
            st.text(doc)
        
        st.subheader("Document Embeddings Visualization")
        fig = visualize_embeddings(question)
        st.plotly_chart(fig)
        
    else:
        st.warning("Please enter a question.")