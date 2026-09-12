"""
Run from the repository root:
    python evaluation/evaluate.py

Requires the backend dependencies and starts no server; it directly loads the search engine.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from search import ChatSearch

queries = json.loads((ROOT/"evaluation/queries.json").read_text(encoding="utf-8"))
engine = ChatSearch()

correct = 0
hard_total = 0
hard_correct = 0

for item in queries:
    results = engine.search(item["query"], top_k=5)
    ids = [r["id"] for r in results]
    ok = item["expected_id"] in ids
    correct += ok
    if item.get("hard"):
        hard_total += 1
        hard_correct += ok
    print(("PASS" if ok else "FAIL"), "|", item["query"], "| expected:", item["expected_id"], "| top:", ids[:1])

print("\n=== CHARCHA EVALUATION ===")
print(f"Total: {len(queries)}")
print(f"Correct in top-5: {correct}/{len(queries)} = {correct/len(queries)*100:.1f}%")
if hard_total:
    print(f"Hard zero-overlap: {hard_correct}/{hard_total} = {hard_correct/hard_total*100:.1f}%")
