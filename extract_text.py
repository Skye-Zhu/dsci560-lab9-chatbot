from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    all_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            all_text.append(f"\n--- Page {page_num} ---\n{text}")
    return "\n".join(all_text)

def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/pdfs/")
        return

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        out_file = OUT_DIR / f"{pdf_file.stem}.txt"
        out_file.write_text(text, encoding="utf-8")
        print(f"Extracted text saved to: {out_file}")

if __name__ == "__main__":
    main()