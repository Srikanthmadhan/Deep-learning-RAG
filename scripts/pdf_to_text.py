import re
from pathlib import Path
from tqdm import tqdm

# PDF
import fitz  # PyMuPDF

# EPUB
from ebooklib import epub
from bs4 import BeautifulSoup

BOOK_DIR = Path(r"D:\Deep learning rag\books")
OUT_RAW = Path(r"D:\Deep learning rag\text_raw")
OUT_CLEAN = Path(r"D:\Deep learning rag\text_clean")

OUT_RAW.mkdir(exist_ok=True)
OUT_CLEAN.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_pdf(path: Path) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)


def extract_epub(path: Path) -> str:
    book = epub.read_epub(path)
    texts = []
    for item in book.get_items():
        if item.get_type() == 9:  # DOCUMENT
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text())
    return "\n".join(texts)


files = list(BOOK_DIR.glob("*"))

for f in tqdm(files, desc="Extracting books"):
    try:
        if f.suffix.lower() == ".pdf":
            raw_text = extract_pdf(f)

        elif f.suffix.lower() == ".epub":
            raw_text = extract_epub(f)

        else:
            continue  # ignore random crap

        # save raw
        raw_path = OUT_RAW / f"{f.stem}.txt"
        raw_path.write_text(raw_text, encoding="utf-8", errors="ignore")

        # clean + save
        clean_path = OUT_CLEAN / f"{f.stem}.txt"
        clean_path.write_text(clean_text(raw_text), encoding="utf-8")

    except Exception as e:
        print(f"Failed on {f.name}: {e}")
