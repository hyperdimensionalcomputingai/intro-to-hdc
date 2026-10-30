# An introduction to HDC with LanceDB and lance-csr

This is a small Python example of hyperdimensional computing (HDC) over a graph.
Four people have visited three cities. We use HDC to find cities matching a description,
then follow exact `VISITED` edges to find who visited them.

- **TorchHD** builds and combines 10,000-dimensional hypervectors.
- **LanceDB** stores graph records, hypervectors, and skyline images together.
- **lance-csr** runs Cypher queries over those same Lance tables.
- **Ollama** supplies local text embeddings for semantic matching.

The scripts are meant to be read and edited. Change a query or setting near the top of a
script, then run it again. There is no command-line option interface or web server.
A new open-source Tauri frontend is planned soon in a separate repository.

## Setup

Install Python 3.13+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com).
From the repository root:

```bash
uv sync
```

For the semantic examples, start Ollama and download the embedding model:

```bash
# Leave this running if Ollama is not already running in the background.
ollama serve
```

In another terminal:

```bash
ollama pull nomic-embed-text
```

The basic HDC and graph-only examples below do not need Ollama.

## Start with binding and bundling

Read [src/hdc_basics.py](src/hdc_basics.py), then run it:

```bash
uv run src/hdc_basics.py
```

A hypervector represents a symbol using many coordinates. Here, atomic symbols use
10,000 deterministic values of `-1` or `+1`. Two operations let us compose them:

| Operation | Meaning | MAP implementation |
|---|---|---|
| Binding: $h_{\mathrm{role}} \otimes h_{\mathrm{value}}$ | Associate a role with a value | Element-wise multiplication |
| Bundling: $h_{\mathrm{name}} \oplus h_{\mathrm{timezone}}$ | Combine several associations | Keep their arithmetic sum |

The script binds the role `name` to `Seattle`, then bundles that association with
`timezone: pacific`. Its assertions show two useful properties:

- Binding again with the same bipolar role recovers the value.
- Subtracting an ingredient from the raw bundle recovers the remaining ingredient.

Keep raw bundle sums. Normalize a composite to bipolar values only before using it as a
binding factor, as the relationship encoder does. Computation uses float32; persisted
hypervectors use float16 to reduce storage. Cast stored vectors back to float32 before
TorchHD operations.

## Run the complete example

With Ollama running:

```bash
uv run src/run_person_location_demo.py
```

This rebuilds `person-location/` from the sample data, adds hypervectors, and asks:

> mountainous cities near the pacific ocean

The script prints ranked candidates with their visitors, followed by all stored graph
paths. Change `QUERY` or `LOCATION_LIMIT` in the script to explore a different question.
The limit counts candidate cities, so one city can return several visitors.

The combined script and the ingestion script **replace the generated dataset** when run.
The query scripts below read it without rebuilding it.

## Work through each stage

Run these in order to see where each part happens:

| Run | What to read or change |
|---|---|
| `uv run src/graph_ingest.py` | Loads the CSV records and skyline images into three LanceDB tables |
| `uv run src/inspect_storage.py` | Reads selected columns, then explicitly loads one image |
| `uv run src/graph_retrieve.py` | Change `QUERY` to find visitors using exact Cypher |
| `uv run src/hdc_encode.py` | Adds hypervectors; `SEMANTIC_WEIGHT` controls the contribution of text evidence |
| `uv run src/hdc_retrieve.py` | Change `QUERY` and `LOCATION_LIMIT` to explore semantic retrieval |

For example, replace the query in `graph_retrieve.py` with:

```python
QUERY = (
    "MATCH (p:Person)-[:VISITED]->(l:Location) "
    "WHERE l.timezone = 'pacific' "
    "RETURN p.name AS person, l.name AS city ORDER BY p.name"
)
```

Then run `uv run src/graph_retrieve.py`. The sample graph returns Maya Chen and Robby Jo,
both connected to Seattle by `VISITED` edges.

For a fuzzy query, edit `hdc_retrieve.py` instead:

```python
QUERY = "concrete jungle with dense skyscrapers"
LOCATION_LIMIT = 1
```

Run `uv run src/hdc_retrieve.py` to inspect the highest-ranked city and its visitors.
This searches the stored hypervectors rather than matching a literal column value.

## How the data flows

```text
CSV graph records ───────────────────────────────→ LanceDB tables
Location evidence → local text embeddings → HDC → Location.hv
Query text        → same embedding/projection    → cosine search
Selected cities   → exact VISITED edges          → people
```

Structural properties use deterministic, hash-derived hypervectors. Location evidence uses
Nomic text embeddings, projected through a seeded Rademacher matrix into bipolar HDC space.
The matrix contains `-1` and `+1`; the projection preserves useful similarity approximately,
without adding semantic information. Documents use `search_document:` and queries use
`search_query:` prefixes.

Each location keeps one full hypervector: its raw structural bundle plus an 8× weighted
semantic term. Eight is a design choice for this small example, not a general optimum.
The projection and model metadata are saved alongside the tables so retrieval uses the
same coordinate system as encoding.

| Table | Graph facts | Stored hypervector |
|---|---|---|
| `Person` | Name, role, identifier | Bundle of bound property pairs |
| `Location` | Name, timezone, region, image | Structural bundle plus weighted semantic evidence |
| `VISITED` | Person and location identifiers | Binding of bipolar person, predicate, and location factors |

Images remain lazy Lance blobs. Encoding and retrieval exclude them from projections;
`inspect_storage.py` shows how to request the bytes explicitly. The graph uses string
identifiers, so lance-csr uses join-based traversal here; CSR indexing requires integer endpoints.

## Explore the code and data

- [src/hdc/](src/hdc/): binding, bundling, projection, encoding, and retrieval.
- [src/graph/](src/graph/): CSV loading, Lance storage, and Cypher queries.
- [src/graph/paths.py](src/graph/paths.py): shared dataset path and local model settings.
- [data/nodes/](data/nodes/) and [data/relationships/](data/relationships/): graph facts.
- [data/features/location_vibes.csv](data/features/location_vibes.csv): hand-authored evidence
  descriptions, standing in for captions a vision model could produce. This demo does not
  infer features from the skyline images.

To change the dataset, edit the CSVs, then rerun ingestion and encoding.

## Checks and limits

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run src/evaluate_semantic_hdc.py
```

The evaluation script needs an encoded dataset and Ollama. It checks nine semantic queries
and three visitor queries. This is a regression check for a three-city example, not a
retrieval benchmark. Cosine scores are ranking signals, not probabilities; the graph
confirms stored visitor edges, not whether a fuzzy description is objectively true.

A 10,000-dimensional float16 hypervector takes 20 KB before compression. The small example
is easy to inspect, but larger datasets need storage and retrieval planning. See
[COMPRESSION_TRICKS.md](COMPRESSION_TRICKS.md) for further experiments.

Dependency compatibility: lance-csr 0.4 requires pylance 13, whose namespace dependency
currently limits this project to LanceDB 0.40. The lockfile records the tested versions.
