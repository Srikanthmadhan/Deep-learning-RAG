import os
import re

BOOK_DIR = r"D:\Deep learning rag\books"

# =========================
# FORCE DELETE KEYWORDS (STRICT)
# =========================
FORCE_PATTERNS = [
    r"\bjava\b",
    r"\bc language\b",
    r"\bc\+\+\b",
    r"\bcpp\b",
    r"\bspring\b",
    r"\bj2ee\b"
]

FORCE_REGEX = re.compile("|".join(FORCE_PATTERNS), re.IGNORECASE)

deleted = []
kept = []

for filename in os.listdir(BOOK_DIR):
    filepath = os.path.join(BOOK_DIR, filename)

    if not os.path.isfile(filepath):
        continue

    name_lower = filename.lower()

    # ONLY delete if C / Java pattern matches
    if FORCE_REGEX.search(name_lower):
        os.remove(filepath)
        deleted.append(filename)
        print(f"DELETED (C/JAVA): {filename}")
    else:
        kept.append(filename)

# =========================
# SUMMARY
# =========================
print("\n========== SUMMARY ==========")
print(f"Deleted C/Java books: {len(deleted)}")
print(f"Kept books: {len(kept)}")
print("================================")
