# rag.py


from transformers import pipeline

# loader.py

import os

def load_documents(path="data"):
    docs = []

    for file in os.listdir(path):
        with open(f"{path}/{file}", "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(text)

    return docs


def main():
    print("Proyecto profesional Python")

    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    clasificador_pro = pipeline("sentiment-analysis", model=model_path)

    textos = [
        "Me encanta este curso de IA, es muy práctico!",
        "El tiempo de espera fue terrible y la comida estaba fría",
        "El hotel estaba bien, nada especial",
        "Increíble experiencia, repetiré sin duda"
     ]

    resultados = clasificador_pro(textos)

    print(f"Usando el modelo: {model_path}\n")
    for texto, resultado in zip(textos, resultados):
        label = resultado['label'].upper()
        score = resultado['score']

    # Lógica de iconos mejorada
    emoji = "✅" if label == "POSITIVE" else "❌" if label == "NEGATIVE" else "😐"

    print(f"{emoji} [{label}: {score:.3f}] {texto}")


    


if __name__ == "__main__":
    main()

