# Punkt 6 - Podcele

## Zdefiniowane podcele

Dla kazdego problemu uzyto dwoch podcelow (pelne stany posrednie).

### Problem A
- Podcel 1: `('c1', 'c2', 'c4', 'c3', 'X', 'c5')`
- Podcel 2: `('X', 'c1', 'c4', 'c3', 'c2', 'c5')`

### Problem B
- Podcel 1: `('c1', 'c2', 'c3', 'c5', 'c4', 'X')`
- Podcel 2: `('c1', 'c2', 'c4', 'c5', 'X', 'c3')`

### Problem C
- Podcel 1: `('c1', 'c2', 'c3', 'c4', 'c5', 'X')`
- Podcel 2: `('c1', 'c3', 'X', 'c4', 'c5', 'c2')`

## Wyniki

Uzyto: `Forward_STRIPS` + `SearcherMPP`, etapowo: start -> podcel1 -> podcel2 -> cel.

### Porownanie: podcele bez heurystyki vs podcele z heurystyka

| Problem | Plan bez h | Plan z h | Czas bez h [s] | Czas z h [s] | Rozszerzone bez h | Rozszerzone z h | Przyspieszenie | Dlugosc planu bez h | Dlugosc planu z h |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | tak | tak | 0.001306 | 0.000231 | 35 | 9 | 5.65x | 6 | 6 |
| B | tak | tak | 0.001014 | 0.000160 | 28 | 7 | 6.34x | 4 | 4 |
| C | tak | tak | 0.000701 | 0.000186 | 22 | 8 | 3.76x | 5 | 5 |

### Problem A - ciag akcji (plan z podcelami i heurystyka)
1. `move_c3_s3_s4`
2. `move_c4_s5_s3`
3. `move_c2_s2_s5`
4. `move_c1_s1_s2`
5. `move_c5_s6_s1`
6. `move_c4_s3_s6`

### Problem B - ciag akcji (plan z podcelami i heurystyka)
1. `move_c5_s6_s4`
2. `move_c3_s3_s6`
3. `move_c4_s5_s3`
4. `move_c3_s6_s5`

### Problem C - ciag akcji (plan z podcelami i heurystyka)
1. `move_c4_s5_s4`
2. `move_c5_s6_s5`
3. `move_c2_s2_s6`
4. `move_c3_s3_s2`
5. `move_c2_s6_s3`
