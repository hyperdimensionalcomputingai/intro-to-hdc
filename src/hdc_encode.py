"""Add HDC columns to the graph tables. Run graph_ingest.py first."""

from graph.paths import DEFAULT_DB_URI
from hdc.encode import encode_hdc

# Demo design choice: give semantic evidence more weight than individual properties.
SEMANTIC_WEIGHT = 8

if __name__ == "__main__":
    encode_hdc(semantic_weight=SEMANTIC_WEIGHT)
    print(f"Added hypervectors and saved projection metadata: {DEFAULT_DB_URI}")
