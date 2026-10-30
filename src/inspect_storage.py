"""Read selected columns, then explicitly load one image."""

import lance

from graph.paths import DEFAULT_DB_URI

locations = lance.dataset(DEFAULT_DB_URI / "Location.lance")
rows = locations.to_table(columns=["id", "name", "timezone"])
print(rows.to_pydict())

# The scan above excludes image bytes and hypervectors.
index = rows.column("id").to_pylist().index("location-seattle")
with locations.take_blobs("image", indices=[index])[0] as blob:
    image_bytes = blob.readall()
print(f"Explicitly loaded Seattle's image: {len(image_bytes)} bytes")
