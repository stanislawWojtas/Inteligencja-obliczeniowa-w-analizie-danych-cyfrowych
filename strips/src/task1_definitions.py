"""
Definicje STRIPS (AIPython) dla zadania 1:
- jedna dziedzina parking
- trzy problemy A/B/C
"""

from collections.abc import Sequence

from aipython.stripsProblem import Planning_problem, Strips, STRIPS_domain

State = dict[str, str]
ProblemKey = str

CARS = ("c1", "c2", "c3", "c4", "c5")
SPOTS = ("s1", "s2", "s3", "s4", "s5", "s6")
VALUES: set[str] = set(CARS) | {"X"}


def arrangement_to_state(arrangement: Sequence[str]) -> State:
    """Np. ('c1','c2','c3','X','c4','c5') -> {'s1':'c1',...,'s6':'c5'}."""
    return {spot: value for spot, value in zip(SPOTS, arrangement)}


def build_parking_domain() -> STRIPS_domain:
    feature_domain_dict: dict[str, set[str]] = {spot: VALUES for spot in SPOTS}
    actions: set[Strips] = set()

    for car in CARS:
        for s_from in SPOTS:
            for s_to in SPOTS:
                if s_from == s_to:
                    continue
                actions.add(
                    Strips(
                        name=f"move_{car}_{s_from}_{s_to}",
                        preconds={s_from: car, s_to: "X"},
                        effects={s_from: "X", s_to: car},
                    )
                )

    return STRIPS_domain(feature_domain_dict, actions)


parking_domain = build_parking_domain()


# Wspólny stan początkowy: {1 2 3 X 4 5}
start_arrangement = ("c1", "c2", "c3", "X", "c4", "c5")
start_state = arrangement_to_state(start_arrangement)


# Problem A: goal {5 1 X 3 2 4}
goal_a_arrangement = ("c5", "c1", "X", "c3", "c2", "c4")
problem_a = Planning_problem(
    parking_domain,
    start_state,
    arrangement_to_state(goal_a_arrangement),
)


# Problem B: goal {1 2 4 5 3 X}
goal_b_arrangement = ("c1", "c2", "c4", "c5", "c3", "X")
problem_b = Planning_problem(
    parking_domain,
    start_state,
    arrangement_to_state(goal_b_arrangement),
)


# Problem C: goal {1 3 2 4 5 X}
goal_c_arrangement = ("c1", "c3", "c2", "c4", "c5", "X")
problem_c = Planning_problem(
    parking_domain,
    start_state,
    arrangement_to_state(goal_c_arrangement),
)


task1_problems = {
    "A": problem_a,
    "B": problem_b,
    "C": problem_c,
}
task1_problems: dict[ProblemKey, Planning_problem]


expected_min_plan_length = {
    "A": 6,
    "B": 4,
    "C": 5,
}
expected_min_plan_length: dict[ProblemKey, int]
