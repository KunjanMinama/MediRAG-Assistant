from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()



def get_llm_chain(retriever):
    
# Create LLM properly
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="chat-completion",
        max_new_tokens=256,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

    # Pass llm to ChatHuggingFace
    model = ChatHuggingFace(llm=llm)

    prompt=PromptTemplate(
        input_variables=["context","question"],
        template="""
        you are **MediBOT**, an aAI-powered assistant trained to help users understand medical documents 
        and health-related questions.

        Your job is to provide clear, accurate, and helpful responses based **only on the provided context**
        
        ---

        **context**:
        {context}

        **user question**:
        {question}

        ---
        
        **Answer**:
        - Respinse in A Calm , factual and respectful tone.
        -use simple explanation when needed.
        -if the context does not contain the n=asnwer, say:"I am sorry, but I couldn't find relevant information in the make up provided documents."
        -do Not make up facts.
        -do not give medical advise or diagnoses.
        """
        )
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
    )

    return rag_chain
    
