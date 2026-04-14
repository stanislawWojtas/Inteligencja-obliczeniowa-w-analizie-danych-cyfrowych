"""
Punkt 2 z TODO:
- forward planning dla problemów A/B/C
- zapis sekwencji działań i czasu
"""

from __future__ import annotations

import time
from typing import Any, Protocol, TypedDict, cast

from aipython.display import Displayable
from aipython.searchMPP import SearcherMPP
from aipython.stripsForwardPlanner import Forward_STRIPS
from task1_definitions import expected_min_plan_length, task1_problems

Displayable.max_display_level = 0


class ActionLike(Protocol):
    name: str


class ArcLike(Protocol):
    action: ActionLike


class PathLike(Protocol):
    arc: ArcLike | None
    initial: PathLike
    cost: int | float


class SolveResult(TypedDict):
    name: str
    found: bool
    elapsed_s: float
    expanded: int
    actions: list[str]
    cost: int | float | None


def extract_actions(path: PathLike) -> list[str]:
    actions: list[str] = []
    current = path
    while current.arc is not None:
        actions.append(current.arc.action.name)
        current = current.initial
    actions.reverse()
    return actions


def solve_problem(name: str) -> SolveResult:
    problem = task1_problems[name]
    search_problem = Forward_STRIPS(problem)
    searcher = SearcherMPP(search_problem)

    t0 = time.perf_counter()
    path_raw: PathLike | None = cast(PathLike | None, searcher.search())
    dt = time.perf_counter() - t0

    if path_raw is None:
        return {
            "name": name,
            "found": False,
            "elapsed_s": dt,
            "expanded": int(cast(Any, searcher).num_expanded),
            "actions": [],
            "cost": None,
        }

    actions = extract_actions(path_raw)
    return {
        "name": name,
        "found": True,
        "elapsed_s": dt,
        "expanded": int(cast(Any, searcher).num_expanded),
        "actions": actions,
        "cost": path_raw.cost,
    }


def build_report(results: list[SolveResult]) -> str:
    lines: list[str] = []
    lines.append("# Punkt 2 - Forward planning")
    lines.append("")
    lines.append("Użyto: `Forward_STRIPS` + `SearcherMPP` z AIPython.")
    lines.append("")

    for result in results:
        name = result["name"]
        lines.append(f"## Problem {name}")
        lines.append(f"- Plan znaleziony: {'tak' if result['found'] else 'nie'}")
        lines.append(f"- Czas: {result['elapsed_s']:.6f} s")
        lines.append(f"- Rozszerzone ścieżki: {result['expanded']}")
        if result["found"]:
            lines.append(f"- Koszt / długość planu: {result['cost']}")
            lines.append(
                f"- Oczekiwana minimalna długość (z punktu 1): {expected_min_plan_length[name]}"
            )
            lines.append("- Ciąg akcji:")
            for idx, action in enumerate(result["actions"], start=1):
                lines.append(f"{idx}. `{action}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    ordered_names = ["A", "B", "C"]
    results = [solve_problem(name) for name in ordered_names]
    report = build_report(results)
    with open("POINT2_FORWARD_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
