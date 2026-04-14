"""
Punkt 8 z TODO:
- trzy dodatkowe problemy D/E/F z podcelami
- kazdy plan (start -> podcel1 -> podcel2 -> cel) ma co najmniej 20 akcji
- rozwiazanie z podcelami bez heurystyki i z heurystyka
"""

from __future__ import annotations

import time
from typing import Any, Protocol, TypedDict, cast

from aipython.display import Displayable
from aipython.searchMPP import SearcherMPP
from aipython.stripsForwardPlanner import Forward_STRIPS
from aipython.stripsProblem import Planning_problem
from task1_definitions import arrangement_to_state, parking_domain, start_state

Displayable.max_display_level = 0

State = dict[str, str]


class ActionLike(Protocol):
    name: str


class ArcLike(Protocol):
    action: ActionLike


class PathLike(Protocol):
    arc: ArcLike | None
    initial: "PathLike"
    cost: int | float


class SegmentResult(TypedDict):
    found: bool
    elapsed_s: float
    expanded: int
    actions: list[str]


class SolveResult(TypedDict):
    name: str
    found: bool
    elapsed_s: float
    expanded: int
    actions: list[str]
    segment_lengths: list[int]


def parking_misplaced_cars_heuristic(state: State, goal: State) -> int:
    return sum(
        1 for spot, target in goal.items() if target != "X" and state[spot] != target
    )


def extract_actions(path: PathLike) -> list[str]:
    actions: list[str] = []
    current = path
    while current.arc is not None:
        actions.append(current.arc.action.name)
        current = current.initial
    actions.reverse()
    return actions


def solve_segment(start: State, goal: State, with_heuristic: bool) -> SegmentResult:
    segment_problem = Planning_problem(parking_domain, start, goal)
    heuristic = parking_misplaced_cars_heuristic if with_heuristic else None
    search_problem = (
        Forward_STRIPS(segment_problem, heuristic)
        if heuristic is not None
        else Forward_STRIPS(segment_problem)
    )
    searcher = SearcherMPP(search_problem)

    t0 = time.perf_counter()
    path_raw: PathLike | None = cast(PathLike | None, searcher.search())
    dt = time.perf_counter() - t0

    if path_raw is None:
        return {
            "found": False,
            "elapsed_s": dt,
            "expanded": int(cast(Any, searcher).num_expanded),
            "actions": [],
        }

    return {
        "found": True,
        "elapsed_s": dt,
        "expanded": int(cast(Any, searcher).num_expanded),
        "actions": extract_actions(path_raw),
    }


# Każdy z problemów D/E/F ma 2 podcele i łączną długość planu minimalnie 21 akcji
# (7 + 7 + 7 segmentów).
problem_specs: dict[str, dict[str, tuple[str, ...] | list[tuple[str, ...]]]] = {
    "D": {
        "goal": ("c2", "c1", "c4", "c3", "X", "c5"),
        "subgoals": [
            ("c2", "c1", "c4", "c5", "c3", "X"),
            ("c1", "c2", "c3", "c4", "c5", "X"),
        ],
    },
    "E": {
        "goal": ("c2", "c1", "c4", "X", "c5", "c3"),
        "subgoals": [
            ("c2", "c1", "c4", "X", "c5", "c3"),
            ("c1", "c2", "c3", "c5", "X", "c4"),
        ],
    },
    "F": {
        "goal": ("c2", "c1", "c3", "c4", "c5", "X"),
        "subgoals": [
            ("c2", "c1", "c5", "c4", "X", "c3"),
            ("c1", "c2", "c4", "c3", "X", "c5"),
        ],
    },
}


def solve_with_subgoals(name: str, with_heuristic: bool) -> SolveResult:
    spec = problem_specs[name]
    subgoals = cast(list[tuple[str, ...]], spec["subgoals"])
    final_goal = cast(tuple[str, ...], spec["goal"])

    goals: list[State] = [arrangement_to_state(x) for x in subgoals]
    goals.append(arrangement_to_state(final_goal))

    current_state: State = start_state.copy()
    all_actions: list[str] = []
    total_elapsed = 0.0
    total_expanded = 0
    segment_lengths: list[int] = []

    for segment_goal in goals:
        segment = solve_segment(current_state, segment_goal, with_heuristic)
        total_elapsed += segment["elapsed_s"]
        total_expanded += segment["expanded"]

        if not segment["found"]:
            return {
                "name": name,
                "found": False,
                "elapsed_s": total_elapsed,
                "expanded": total_expanded,
                "actions": all_actions,
                "segment_lengths": segment_lengths,
            }

        all_actions.extend(segment["actions"])
        segment_lengths.append(len(segment["actions"]))
        current_state = segment_goal.copy()

    return {
        "name": name,
        "found": True,
        "elapsed_s": total_elapsed,
        "expanded": total_expanded,
        "actions": all_actions,
        "segment_lengths": segment_lengths,
    }


def build_report(
    plain_results: dict[str, SolveResult], heuristic_results: dict[str, SolveResult]
) -> str:
    lines: list[str] = []
    lines.append("# Punkt 8 - Dodatkowe problemy z podcelami")
    lines.append("")
    lines.append("## Definicje dodatkowych problemow")
    lines.append("")
    lines.append(
        "Wszystkie problemy startuja ze stanu `('c1','c2','c3','X','c4','c5')` i maja po 2 podcele."
    )
    lines.append("")

    for name in ["D", "E", "F"]:
        spec = problem_specs[name]
        subgoals = cast(list[tuple[str, ...]], spec["subgoals"])
        goal = cast(tuple[str, ...], spec["goal"])
        lines.append(f"### Problem {name}")
        lines.append(f"- Podcel 1: `{subgoals[0]}`")
        lines.append(f"- Podcel 2: `{subgoals[1]}`")
        lines.append(f"- Cel koncowy: `{goal}`")
        lines.append("")

    lines.append("## Wyniki")
    lines.append("")
    lines.append("Uzyto: `Forward_STRIPS` + `SearcherMPP`, etapowo: start -> podcel1 -> podcel2 -> cel.")
    lines.append("")
    lines.append(
        "| Problem | Plan bez h | Plan z h | Czas bez h [s] | Czas z h [s] | Rozszerzone bez h | Rozszerzone z h | Dlugosc bez h | Dlugosc z h | Segmenty z h | Warunek >=20 |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")

    for name in ["D", "E", "F"]:
        plain = plain_results[name]
        heur = heuristic_results[name]
        len_plain = len(plain["actions"]) if plain["found"] else None
        len_heur = len(heur["actions"]) if heur["found"] else None
        segments = (
            " + ".join(str(x) for x in heur["segment_lengths"])
            if heur["found"]
            else "-"
        )
        at_least_20 = "tak" if len_heur is not None and len_heur >= 20 else "nie"

        lines.append(
            f"| {name} | {'tak' if plain['found'] else 'nie'} | {'tak' if heur['found'] else 'nie'} | "
            f"{plain['elapsed_s']:.6f} | {heur['elapsed_s']:.6f} | "
            f"{plain['expanded']} | {heur['expanded']} | "
            f"{len_plain if len_plain is not None else '-'} | "
            f"{len_heur if len_heur is not None else '-'} | "
            f"{segments} | {at_least_20} |"
        )
    lines.append("")

    for name in ["D", "E", "F"]:
        lines.append(f"### Problem {name} - ciag akcji (plan z podcelami i heurystyka)")
        heur = heuristic_results[name]
        if heur["found"]:
            for idx, action in enumerate(heur["actions"], start=1):
                lines.append(f"{idx}. `{action}`")
        else:
            lines.append("Brak planu.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    plain_results = {
        name: solve_with_subgoals(name, with_heuristic=False)
        for name in ["D", "E", "F"]
    }
    heuristic_results = {
        name: solve_with_subgoals(name, with_heuristic=True)
        for name in ["D", "E", "F"]
    }

    report = build_report(plain_results, heuristic_results)
    with open("POINT8_EXTENDED_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
