"""Exercise the replacement engine against real Lance tables, including lazy blobs."""

from pathlib import Path

from graph.api import FilterSpec, QuerySpec, RenderSpec, assemble, get_image, schema_payload
from graph.ingest import ingest_graph
from graph.retrieve import city_query, execute_query, timezone_query, visited_query


def test_graph_queries_and_visualizer(tmp_path: Path) -> None:
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

    schema = schema_payload(db_uri)
    location = next(node for node in schema["nodes"] if node["label"] == "Location")
    assert "image" in location["assets"]
    assert "image" not in location["properties"]
    result = assemble(
        QuerySpec(
            edge="VISITED",
            render={"Person": RenderSpec(label="name"), "Location": RenderSpec(label="name")},
            filters=[FilterSpec(label="Location", prop="name", op="=", value="Seattle")],
        ),
        db_uri,
    )
    assert len(result["links"]) == 2
    assert {node["label"] for node in result["nodes"]} == {"Maya Chen", "Robby Jo", "Seattle"}
    assert get_image("Location", "location-seattle", db_uri).body
