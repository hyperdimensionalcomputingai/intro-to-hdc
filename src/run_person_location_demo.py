"""Build the small dataset and run HDC retrieval followed by graph traversal."""

from graph.ingest import ingest_graph
from graph.retrieve import execute_query, visited_query
from hdc.encode import encode_hdc
from hdc.retrieve import hdc_fuzzy_paths

QUERY = "mountainous cities near the pacific ocean"
LOCATION_LIMIT = 2

if __name__ == "__main__":
    # Rebuilds person-location/ from the example CSVs on every run.
    ingest_graph()
    encode_hdc()

    print(f"Question: {QUERY}")
    print("HDC candidates, expanded through lance-csr:")
    for row in hdc_fuzzy_paths(QUERY, limit=LOCATION_LIMIT):
        print(f"  {row['person']} -> {row['city']} (score={row['score']:.3f})")

    print("All stored VISITED paths:")
    for row in execute_query(visited_query()).to_pylist():
        print(f"  {row['person']} -> {row['city']}")
