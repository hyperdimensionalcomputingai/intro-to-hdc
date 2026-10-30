from __future__ import annotations

from evaluation.semantic import (
    DEFAULT_PATH_QUERY_SET,
    DEFAULT_QUERY_SET,
    EvaluationResult,
    evaluate,
)
from graph.paths import DEFAULT_DB_URI, DEFAULT_OLLAMA_HOST

LOCATION_LIMIT = 3  # Evaluate all three cities in this tiny corpus.


def print_results(title: str, results: list[EvaluationResult]) -> None:
    print(title)
    for result in results:
        status = "PASS" if result.passes else "FAIL"
        ranking = ", ".join(f"{city}={score:.3f}" for city, score in result.ranked_cities)
        print(
            f"{status} expected={result.expected_city!r} "
            f"rank={result.expected_rank} "
            f"query={result.query!r}"
        )
        print(f"  {ranking}")
        if result.expected_people:
            visitors = " | ".join(result.people_by_city.get(result.expected_city, ()))
            print(f"  visitors={visitors}")


def main() -> int:
    semantic_results = evaluate(
        DEFAULT_QUERY_SET,
        DEFAULT_DB_URI,
        DEFAULT_OLLAMA_HOST,
        LOCATION_LIMIT,
    )
    path_results = evaluate(
        DEFAULT_PATH_QUERY_SET,
        DEFAULT_DB_URI,
        DEFAULT_OLLAMA_HOST,
        LOCATION_LIMIT,
        require_expected_people=True,
    )
    print_results("Semantic city ranking", semantic_results)
    print()
    print_results("Semantic ranking + VISITED traversal", path_results)

    top_one = sum(result.expected_rank == 1 for result in semantic_results)
    path_passes = sum(result.passes for result in path_results)
    mrr = sum(result.reciprocal_rank for result in semantic_results) / len(semantic_results)
    print()
    print(
        f"Semantic top-1 accuracy: {top_one}/{len(semantic_results)} "
        f"({top_one / len(semantic_results):.1%})"
    )
    print(f"Semantic mean reciprocal rank: {mrr:.3f}")
    print(
        f"VISITED path checks: {path_passes}/{len(path_results)} "
        f"({path_passes / len(path_results):.1%})"
    )
    return 0 if all(result.passes for result in [*semantic_results, *path_results]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
