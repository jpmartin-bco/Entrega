from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# rag.py

from langchain_community.embeddings import HuggingFaceEmbeddings

def create_db(texts):
    splitter = CharacterTextSplitter(chunk_size=500)

    documents = []
    for text in texts:
        documents.extend(splitter.split_text(text))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_texts(documents, embeddings)

    return db

def ask(db, question):

    docs = db.similarity_search_with_score(question, k=3)

    best_score = docs[0][1]

    if best_score > 1.0:  # ajusta este umbral
        return "No hay información suficiente en los documentos para responder esta pregunta."

    context = "\n\n".join([doc[0].page_content for doc in docs])

    return context