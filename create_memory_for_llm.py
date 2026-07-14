from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

documents = PyPDFLoader("data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf").load()

text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

DB_FAISS_PATH = "vectorstore/faiss_index"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
 