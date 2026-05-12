# app.py

import gradio as gr
from .rag import create_db, ask
from .loader import load_documents


docs = load_documents()
db = create_db(docs)


def chat(question):
    context = ask(db, question)

    return f"Respuesta basada en documentos:\n\n{context}"


gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="RAG Chatbot"
).launch()