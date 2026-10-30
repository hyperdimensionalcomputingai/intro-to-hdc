"""Load the example CSV records and images into LanceDB."""

from graph.ingest import ingest_graph
from graph.paths import DEFAULT_DB_URI

if __name__ == "__main__":
    # Rebuilds the demo tables; rerun hdc_encode.py afterwards.
    ingest_graph()
    print(f"Wrote graph tables: {DEFAULT_DB_URI}")
