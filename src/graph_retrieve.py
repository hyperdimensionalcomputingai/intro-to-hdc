"""Find visitors using exact graph facts, without embeddings."""

from graph.retrieve import city_query, execute_query

# Change the city, or replace QUERY with your own Cypher string.
QUERY = city_query("Seattle")

if __name__ == "__main__":
    for row in execute_query(QUERY).to_pylist():
        print(row)
