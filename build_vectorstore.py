from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

TEXT_DIR = Path("data/extracted")
VECTOR_DIR = "vectorstore/faiss_index"

#
def load_all_text():
    all_docs = []
    for txt_file in TEXT_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        all_docs.append((txt_file.name, text))
    return all_docs


def split_documents(docs):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    texts = []
    metadatas = []

    for filename, text in docs:
        text_chunks = splitter.split_text(text)
        for i, chunk in enumerate(text_chunks):
            texts.append(chunk)
            metadatas.append({
                "source": filename,
                "chunk_id": i
            })

    return texts, metadatas


def main():
    docs = load_all_text()
    texts, metadatas = split_documents(docs)

    print(f"Loaded {len(docs)} text files")
    print(f"Creating embeddings for {len(texts)} chunks...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    Path("vectorstore").mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTOR_DIR)

    print(f"Vector store saved to: {VECTOR_DIR}")


if __name__ == "__main__":
    main()