from langchain_community.vectorstores import FAISS
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
        model="distilgpt2",
        max_new_tokens=100
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
    main()