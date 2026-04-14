"""
Punkt 3 z TODO:
- heurystyka dla problemu parkingowego STRIPS
- rozwiazanie problemow A/B/C z heurystyka i bez
- zapis czasu oraz porownania
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
    initial: "PathLike"
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


def parking_misplaced_cars_heuristic(
    state: dict[str, str], goal: dict[str, str]
) -> int:
    """
    Admissible lower bound dla problemu parkingowego:
    liczba samochodow, ktore nie stoja na docelowych miejscach.

    Kazda akcja przesuwa dokladnie jeden samochod, wiec w pojedynczym kroku
    da sie poprawic pozycje co najwyzej jednego samochodu.
    """
    return sum(
        1 for spot, target in goal.items() if target != "X" and state[spot] != target
    )


def solve_problem(name: str, with_heuristic: bool) -> SolveResult:
    problem = task1_problems[name]
    heuristic = parking_misplaced_cars_heuristic if with_heuristic else None
    search_problem = (
        Forward_STRIPS(problem, heuristic)
        if heuristic is not None
        else Forward_STRIPS(problem)
    )
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


def build_report(
    plain_results: dict[str, SolveResult], heuristic_results: dict[str, SolveResult]
) -> str:
    lines: list[str] = []
    lines.append("# Punkt 3 - Heurystyka")
    lines.append("")
    lines.append("## Zaproponowana heurystyka")
    lines.append("")
    lines.append(
        "Heurystyka `h(state) = liczba samochodow stojacych na niewlasciwych miejscach` "
        "(bez liczenia pustego miejsca `X`)."
    )
    lines.append(
        "W pojedynczej akcji przesuwany jest dokladnie jeden samochod, wiec ta wartosc "
        "jest dolnym ograniczeniem liczby krokow do celu (heurystyka admissible)."
    )
    lines.append(
        "Jest pomocna, bo preferuje stany, w ktorych wiecej samochodow juz pokrywa sie z celem, "
        "co ogranicza eksploracje stanow malo obiecujacych."
    )
    lines.append("")
    lines.append("## Wyniki")
    lines.append("")
    lines.append("Uzyto: `Forward_STRIPS` + `SearcherMPP`.")
    lines.append("")

    lines.append("### Porownanie: bez heurystyki vs z heurystyka")
    lines.append("")
    lines.append(
        "| Problem | Plan bez h | Plan z h | Czas bez h [s] | Czas z h [s] | Rozszerzone bez h | Rozszerzone z h | Przyspieszenie | Dlugosc planu z h | Min. dlugosc (pkt 1) |"
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
        plan_len = heur["cost"] if heur["found"] else "-"
        min_len = expected_min_plan_length[name]
        lines.append(
            f"| {name} | {'tak' if plain['found'] else 'nie'} | {'tak' if heur['found'] else 'nie'} | "
            f"{plain['elapsed_s']:.6f} | {heur['elapsed_s']:.6f} | {plain['expanded']} | "
            f"{heur['expanded']} | "
            f"{f'{speedup:.2f}x' if speedup is not None else '-'} | {plan_len} | {min_len} |"
        )
    lines.append("")

    for name in ["A", "B", "C"]:
        heur = heuristic_results[name]
        lines.append(f"### Problem {name} - ciag akcji (plan z heurystyka)")
        if heur["found"]:
            for idx, action in enumerate(heur["actions"], start=1):
                lines.append(f"{idx}. `{action}`")
        else:
            lines.append("Brak planu.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    plain_results = {
        name: solve_problem(name, with_heuristic=False) for name in ["A", "B", "C"]
    }
    heuristic_results = {
        name: solve_problem(name, with_heuristic=True) for name in ["A", "B", "C"]
    }
    report = build_report(plain_results, heuristic_results)
    with open("POINT3_HEURISTIC_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
