"""Exercise the replacement engine against real Lance tables, including lazy blobs."""

from pathlib import Path

import lance

from graph.ingest import ingest_graph
from graph.paths import PROJECT_ROOT
from graph.retrieve import city_query, execute_query, timezone_query, visited_query


def test_graph_queries_and_lazy_images(tmp_path: Path) -> None:
    db_uri = tmp_path / "graph"
    ingest_graph(db_uri)

    seattle = execute_query(city_query("Seattle"), db_uri).to_pylist()
    assert seattle == [
        {"person": "Maya Chen", "city": "Seattle"},
        {"person": "Robby Jo", "city": "Seattle"},
    ]
    assert len(execute_query(visited_query(), db_uri)) == 4
    assert len(execute_query(timezone_query("pacific"), db_uri)) == 2
    assert execute_query(city_query("missing"), db_uri).num_rows == 0

    locations = lance.dataset(db_uri / "Location.lance")
    catalog = locations.to_table(columns=["id", "name"])
    assert catalog.column_names == ["id", "name"]
    index = catalog.column("id").to_pylist().index("location-seattle")
    with locations.take_blobs("image", indices=[index])[0] as blob:
        assert blob.readall() == (PROJECT_ROOT / "img" / "seattle.jpg").read_bytes()
