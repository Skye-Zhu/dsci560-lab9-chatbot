'''from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

VECTOR_DIR = "vectorstore/faiss_index"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def build_prompt(context, question):
    return f"""You are a helpful Q&A assistant.
Answer the user's question using only the context below.
If the answer is not in the context, say you could not find it in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""


def main():
    print("Loading vector store...")
    vectorstore = load_vectorstore()

    print("Loading local language model...")
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=128
    )

    print("\nChatbot is ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        print("\n[DEBUG] Retrieved Context:\n", context[:1000], "\n")

        prompt = build_prompt(context, question)
        result = generator(prompt, do_sample=False)

        full_output = result[0]["generated_text"]

        if "Answer:" in full_output:
            answer = full_output.split("Answer:")[-1].strip()
        else:
            answer = full_output.strip()

        print("\nBot:", answer)
        print("\nTop retrieved chunks:")
        for doc in docs:
            print(doc.metadata)

        print("-" * 60)


if __name__ == "__main__":
    main()'''

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

VECTOR_DIR = "vectorstore/faiss_index"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def extract_short_answer(question, text):
    q = question.lower()

    if "instructor" in q:
        match = re.search(r"Instructor:\s*(.+)", text)
        if match:
            return match.group(1).strip()

    if "goal" in q:
        match = re.search(r"The Goal:\s*(.+)", text)
        if match:
            return match.group(1).strip()

    return text

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_best_sentences(question, docs, embeddings, top_n=2):
    sentences = []

    for doc in docs:
        sentences.extend(split_into_sentences(doc.page_content))

    if not sentences:
        return "I could not find relevant information in the PDF."

    question_emb = embeddings.embed_query(question)
    sentence_embs = embeddings.embed_documents(sentences)

    from sklearn.metrics.pairwise import cosine_similarity

    scores = cosine_similarity([question_emb], sentence_embs)[0]

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    best_sentences = [s for s, _ in ranked[:top_n]]

    return " ".join(best_sentences)


def main():
    print("Loading vector store...")
    vectorstore = load_vectorstore()

    print("\nChatbot is ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        docs = vectorstore.similarity_search(question, k=3)

        if docs:
            raw_answer = docs[0].page_content
            answer = extract_short_answer(question, raw_answer)
        else:
            answer = "I could not find relevant information in the PDF."

        print("\nBot:")
        print(answer)

        print("\nTop retrieved chunks:")
        for doc in docs:
            print(doc.metadata)

        print("-" * 60)


if __name__ == "__main__":
    main()