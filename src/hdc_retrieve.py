"""Rank cities by meaning, then find their visitors through exact graph edges."""

from hdc.retrieve import hdc_fuzzy_paths

# Edit these and rerun the script to explore other descriptions.
QUERY = "mountainous cities near the pacific ocean"
LOCATION_LIMIT = 2  # Candidate cities, not people: a city can have several visitors.

if __name__ == "__main__":
    for row in hdc_fuzzy_paths(QUERY, limit=LOCATION_LIMIT):
        print({**row, "score": round(row["score"], 3)})
