import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def get_llm_chain(retriever):
    # ✅ Moved here
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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

        If the answer exists anywhere in the context, you MUST provide it.
        Do not say you cannot find it if the information is present.
        
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

    # -------------------------------
    # CASE 1: No retriever (You are using reranking)
    # -------------------------------
    if retriever is None:
        rag_chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | prompt
            | model
        )
        return rag_chain

    # -------------------------------
    # CASE 2: Normal RAG with retriever
    # -------------------------------
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
    )

    return rag_chain
    
 

