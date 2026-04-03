# Zadanie 1 - przykładowe problemy STRIPS

## Wybrana dziedzina

`parking` (jedna pusta pozycja `X`, ruch: przeniesienie jednego auta na puste miejsce).

Źródło domeny i przykładowych instancji:

- <https://gist.github.com/primaryobjects/5ddc68b094d5affc492d2072e85be05b>

## Dlaczego warunek "co najmniej 50 stanów" jest spełniony

Dla 6 miejsc (`s1..s6`), 5 aut (`c1..c5`) i 1 pustego miejsca liczba stanów to liczba permutacji 6 symboli (`c1..c5` i `X`):

`6! = 720` stanów.

Każdy z poniższych problemów działa w tej samej przestrzeni stanów (720 > 50).

## Problem A (z gotowego przykładu)

- Start: `{1 2 3 X 4 5}`
- Goal: `{5 1 X 3 2 4}`
- Minimalny plan: 6 akcji

Plan:

1. `move c1 s1 s4`
2. `move c5 s6 s1`
3. `move c4 s5 s6`
4. `move c2 s2 s5`
5. `move c1 s4 s2`
6. `move c3 s3 s4`

(Uwaga: to odpowiada `problem2.txt` z powyższego źródła.)

## Problem B

- Start: `{1 2 3 X 4 5}`
- Goal: `{1 2 4 5 3 X}`
- Minimalny plan: 4 akcje

Plan:

1. `move c3 s3 s4`
2. `move c4 s5 s3`
3. `move c3 s4 s5`
4. `move c5 s6 s4`

## Problem C

- Start: `{1 2 3 X 4 5}`
- Goal: `{1 3 2 4 5 X}`
- Minimalny plan: 5 akcji

Plan:

1. `move c2 s2 s4`
2. `move c3 s3 s2`
3. `move c2 s4 s3`
4. `move c4 s5 s4`
5. `move c5 s6 s5`
