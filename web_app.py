from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

app = Flask(__name__)

VECTOR_DIR = "vectorstore/faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    VECTOR_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

chat_history = []


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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form["question"].strip()
        docs = vectorstore.similarity_search(question, k=3)

        if docs:
            raw_answer = docs[0].page_content
            answer = extract_short_answer(question, raw_answer)
        else:
            answer = "I could not find relevant information in the PDF."

        chat_history.append(("You", question))
        chat_history.append(("Bot", answer))

    return render_template("index.html", chat_history=chat_history)


if __name__ == "__main__":
    app.run(debug=True)