from bm25s import BM25
import inspect

print("BM25 Loaded From:")
print(inspect.getfile(BM25))

print("\nBM25 Methods:")
print([m for m in dir(BM25) if not m.startswith("_")])
