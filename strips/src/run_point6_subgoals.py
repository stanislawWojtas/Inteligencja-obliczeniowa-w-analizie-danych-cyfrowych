"""
Punkt 6 z TODO:
- podcele dla problemow A/B/C (co najmniej 2 na problem)
- rozwiazanie z podcelami bez heurystyki i z heurystyka
- zapis czasu, liczby rozszerzen i ciagu akcji
"""

from __future__ import annotations

import time
from typing import Any, Protocol, TypedDict, cast

from aipython.display import Displayable
from aipython.searchMPP import SearcherMPP
from aipython.stripsForwardPlanner import Forward_STRIPS
from aipython.stripsProblem import Planning_problem
from task1_definitions import arrangement_to_state, task1_problems

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
    segments: list[SegmentResult]


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


def solve_segment(
    problem_name: str, start_state: State, goal_state: State, with_heuristic: bool
) -> SegmentResult:
    segment_problem = Planning_problem(
        task1_problems[problem_name].prob_domain, start_state, goal_state
    )
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


subgoal_arrangements: dict[str, list[tuple[str, ...]]] = {
    "A": [
        ("c1", "c2", "c4", "c3", "X", "c5"),
        ("X", "c1", "c4", "c3", "c2", "c5"),
    ],
    "B": [
        ("c1", "c2", "c3", "c5", "c4", "X"),
        ("c1", "c2", "c4", "c5", "X", "c3"),
    ],
    "C": [
        ("c1", "c2", "c3", "c4", "c5", "X"),
        ("c1", "c3", "X", "c4", "c5", "c2"),
    ],
}


def solve_with_subgoals(name: str, with_heuristic: bool) -> SolveResult:
    problem = task1_problems[name]
    goals: list[State] = [
        arrangement_to_state(arrangement) for arrangement in subgoal_arrangements[name]
    ]
    goals.append(cast(State, problem.goal))

    current_state = cast(State, problem.initial_state.copy())
    all_actions: list[str] = []
    segments: list[SegmentResult] = []
    total_elapsed = 0.0
    total_expanded = 0

    for segment_goal in goals:
        segment = solve_segment(name, current_state, segment_goal, with_heuristic)
        segments.append(segment)
        total_elapsed += segment["elapsed_s"]
        total_expanded += segment["expanded"]

        if not segment["found"]:
            return {
                "name": name,
                "found": False,
                "elapsed_s": total_elapsed,
                "expanded": total_expanded,
                "actions": all_actions,
                "segments": segments,
            }

        all_actions.extend(segment["actions"])
        current_state = segment_goal.copy()

    return {
        "name": name,
        "found": True,
        "elapsed_s": total_elapsed,
        "expanded": total_expanded,
        "actions": all_actions,
        "segments": segments,
    }


def build_report(
    plain_results: dict[str, SolveResult], heuristic_results: dict[str, SolveResult]
) -> str:
    lines: list[str] = []
    lines.append("# Punkt 6 - Podcele")
    lines.append("")
    lines.append("## Zdefiniowane podcele")
    lines.append("")
    lines.append("Dla kazdego problemu uzyto dwoch podcelow (pelne stany posrednie).")
    lines.append("")
    for name in ["A", "B", "C"]:
        lines.append(f"### Problem {name}")
        for idx, arrangement in enumerate(subgoal_arrangements[name], start=1):
            lines.append(f"- Podcel {idx}: `{arrangement}`")
        lines.append("")

    lines.append("## Wyniki")
    lines.append("")
    lines.append("Uzyto: `Forward_STRIPS` + `SearcherMPP`, etapowo: start -> podcel1 -> podcel2 -> cel.")
    lines.append("")
    lines.append("### Porownanie: podcele bez heurystyki vs podcele z heurystyka")
    lines.append("")
    lines.append(
        "| Problem | Plan bez h | Plan z h | Czas bez h [s] | Czas z h [s] | Rozszerzone bez h | Rozszerzone z h | Przyspieszenie | Dlugosc planu bez h | Dlugosc planu z h |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ["A", "B", "C"]:
        plain = plain_results[name]
        heur = heuristic_results[name]
        speedup = (
            plain["elapsed_s"] / heur["elapsed_s"]
            if plain["found"] and heur["found"] and heur["elapsed_s"] > 0
            else None
        )
        lines.append(
            f"| {name} | {'tak' if plain['found'] else 'nie'} | {'tak' if heur['found'] else 'nie'} | "
            f"{plain['elapsed_s']:.6f} | {heur['elapsed_s']:.6f} | "
            f"{plain['expanded']} | {heur['expanded']} | "
            f"{f'{speedup:.2f}x' if speedup is not None else '-'} | "
            f"{len(plain['actions']) if plain['found'] else '-'} | "
            f"{len(heur['actions']) if heur['found'] else '-'} |"
        )
    lines.append("")

    for name in ["A", "B", "C"]:
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
        name: solve_with_subgoals(name, with_heuristic=False) for name in ["A", "B", "C"]
    }
    heuristic_results = {
        name: solve_with_subgoals(name, with_heuristic=True) for name in ["A", "B", "C"]
    }
    report = build_report(plain_results, heuristic_results)
    with open("POINT6_SUBGOALS_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
