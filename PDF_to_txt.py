from pathlib import Path
from pypdf import PdfReader

PAPERS_DIR = Path("RAG\Literature")
OUTPUT_DIR = Path("RAG\extracted_text")
OUTPUT_DIR.mkdir(exist_ok=True)

pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found. Place both papers inside the papers folder."
    )

for pdf_path in pdf_files:
    reader = PdfReader(str(pdf_path))
    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        extracted_pages.append(
            f"\n--- PAGE {page_number} ---\n{text}"
        )

    full_text = "\n".join(extracted_pages)

    output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
    output_path.write_text(full_text, encoding="utf-8")

    print(f"\nPaper: {pdf_path.name}")
    print(f"Pages: {len(reader.pages)}")
    print(f"Characters extracted: {len(full_text)}")
    print(f"Saved to: {output_path}")

    if len(full_text.strip()) < 500:
        print("WARNING: Very little text extracted; inspect the PDF.")
