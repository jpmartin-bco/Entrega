# loader.py

import os

def load_documents(path="data"):
    docs = []

    for file in os.listdir(path):
        with open(f"{path}/{file}", "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(text)

    return docs