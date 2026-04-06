
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
        You are **MediBOT**, a precise medical document assistant.

IMPORTANT: All chunks from the same file belong to the SAME patient.
Use the "patient_file" field in sources to group information correctly.
CONVERSATION HISTORY:
{chat_history}

CONTEXT FROM UPLOADED DOCUMENTS:
{context}

STRICT RULES:
1. Chunks with same source file = same patient
2. ONLY use information explicitly present in the context
3. NEVER assume a chunk is a different patient just because name isn't repeated
4. If answer is NOT in context say exactly: "I could not find this information in the uploaded documents."
5. NEVER guess or use general medical knowledge
6. NEVER say "typically", "usually", "generally"
7. Always mention source/patient file when answering
8. Respond naturally and clearly

USER QUESTION: {question}
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
    
 
