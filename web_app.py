from flask import Flask, render_template, request, redirect, url_for
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from pathlib import Path
import re
import os

app = Flask(__name__)

UPLOAD_FOLDER = Path("data/pdfs")
EXTRACTED_FOLDER = Path("data/extracted")
VECTOR_DIR = "vectorstore/faiss_index"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)
Path("vectorstore").mkdir(parents=True, exist_ok=True)

chat_history = []
status_message = "No PDF uploaded yet."

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    all_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            all_text.append(f"\n--- Page {page_num} ---\n{text}")

    return "\n".join(all_text)


def process_uploaded_pdf(pdf_path: Path):
    text = extract_text_from_pdf(pdf_path)
    txt_path = EXTRACTED_FOLDER / f"{pdf_path.stem}.txt"
    txt_path.write_text(text, encoding="utf-8")
    return txt_path


def load_all_text():
    all_docs = []
    for txt_file in EXTRACTED_FOLDER.glob("*.txt"):
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


def rebuild_vectorstore():
    global vectorstore

    docs = load_all_text()
    if not docs:
        vectorstore = None
        return False

    texts, metadatas = split_documents(docs)

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    vectorstore.save_local(VECTOR_DIR)
    return True


def extract_short_answer(question, text):
    q = question.lower()

    if "instructor" in q:
        match = re.search(r"Instructor:\s*(.+)", text)
        if match:
            return f"The instructor is {match.group(1).strip()}."

    if "goal" in q:
        match = re.search(r"The Goal:\s*(.+)", text)
        if match:
            return f"The goal of the project is {match.group(1).strip().lower()}."

    return None


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

    scores = cosine_similarity([question_emb], sentence_embs)[0]
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    best_sentences = [s for s, _ in ranked[:top_n]]
    answer = " ".join(best_sentences)

    answer = answer.replace("1) The Goal:", "The goal of the project is")
    return answer


@app.route("/", methods=["GET", "POST"])
def index():
    global status_message

    if request.method == "POST":
        if "pdf_file" in request.files and request.files["pdf_file"].filename:
            pdf_file = request.files["pdf_file"]

            if pdf_file.filename.lower().endswith(".pdf"):
                save_path = UPLOAD_FOLDER / pdf_file.filename
                pdf_file.save(save_path)

                process_uploaded_pdf(save_path)
                success = rebuild_vectorstore()

                if success:
                    status_message = f"Uploaded and indexed: {pdf_file.filename}"
                else:
                    status_message = "Upload succeeded, but no text was extracted."
            else:
                status_message = "Please upload a valid PDF file."

            return redirect(url_for("index"))

        question = request.form.get("question", "").strip()

        if question:
            if vectorstore is None:
                answer = "Please upload and index a PDF first."
                chat_history.append(("You", question))
                chat_history.append(("Bot", answer))
            else:
                docs = vectorstore.similarity_search(question, k=3)

                if docs:
                    raw_answer = docs[0].page_content
                    short_answer = extract_short_answer(question, raw_answer)

                    if short_answer:
                        answer = short_answer
                    else:
                        answer = extract_best_sentences(question, docs, embeddings, top_n=2)

                    source = docs[0].metadata.get("source", "unknown")
                    chunk_id = docs[0].metadata.get("chunk_id", "unknown")
                    #answer += f"\n\nSource: {source} | Chunk: {chunk_id}"
                    answer += f"<br><br><small>Source: {source} | Chunk: {chunk_id}</small>"
                else:
                    answer = "I could not find relevant information in the PDF."

                chat_history.append(("You", question))
                chat_history.append(("Bot", answer))

    return render_template(
        "index.html",
        chat_history=chat_history,
        status_message=status_message
    )


if __name__ == "__main__":
    # 如果本地已有 txt，可以启动时自动建一次 index
    if any(EXTRACTED_FOLDER.glob("*.txt")):
        rebuild_vectorstore()
        status_message = "Existing extracted files loaded."
    app.run(debug=True)