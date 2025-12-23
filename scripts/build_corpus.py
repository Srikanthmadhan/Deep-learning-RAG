from pathlib import Path
from tqdm import tqdm

CLEAN_DIR = Path(r"D:\Deep learning rag\text_clean")
OUT_FILE = Path(r"D:\Deep learning rag\corpus.txt")

files = sorted(CLEAN_DIR.glob("*.txt"))

with OUT_FILE.open("w", encoding="utf-8") as out:
    for f in tqdm(files, desc="Merging corpus"):
        out.write("\n\n")
        out.write("=" * 80 + "\n")
        out.write(f"BOOK: {f.stem}\n")
        out.write("=" * 80 + "\n\n")

        text = f.read_text(encoding="utf-8", errors="ignore")
        out.write(text)
        out.write("\n\n")

print(f"Corpus built: {OUT_FILE}")
