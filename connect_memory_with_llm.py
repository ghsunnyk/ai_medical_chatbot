import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env file")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.6,
    huggingfacehub_api_token=HF_TOKEN
)

model = ChatHuggingFace(llm=llm)


DB_FAISS_PATH = "vectorstore/faiss_index"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 3})

prompt_template = """
You are a medical assistant. 

You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.

If you don't know the answer, just say "I don't know". Do not try to make up an answer.

Context: {context}
Question: {question}

Answer in Markdown:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


chain = prompt | model | StrOutputParser()


def ask_question(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    return chain.invoke({ "context": context, "question": question })


while True:
    question = input("\nAsk a medical question (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    answer = ask_question(question)
    print("\nAnswer:\n", answer)