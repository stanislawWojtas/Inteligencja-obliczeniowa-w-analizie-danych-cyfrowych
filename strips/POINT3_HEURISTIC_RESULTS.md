# Punkt 3 - Heurystyka

## Zaproponowana heurystyka

Heurystyka `h(state) = liczba samochodow stojacych na niewlasciwych miejscach` (bez liczenia pustego miejsca `X`).
W pojedynczej akcji przesuwany jest dokladnie jeden samochod, wiec ta wartosc jest dolnym ograniczeniem liczby krokow do celu (heurystyka admissible).
Jest pomocna, bo preferuje stany, w ktorych wiecej samochodow juz pokrywa sie z celem, co ogranicza eksploracje stanow malo obiecujacych.

## Wyniki

Uzyto: `Forward_STRIPS` + `SearcherMPP`.

### Porownanie: bez heurystyki vs z heurystyka

| Problem | Plan bez h | Plan z h | Czas bez h [s] | Czas z h [s] | Rozszerzone bez h | Rozszerzone z h | Przyspieszenie | Dlugosc planu z h | Min. dlugosc (pkt 1) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | tak | tak | 0.082553 | 0.000303 | 532 | 7 | 272.86x | 6 | 6 |
| B | tak | tak | 0.019016 | 0.000168 | 204 | 5 | 112.91x | 4 | 4 |
| C | tak | tak | 0.056221 | 0.000204 | 307 | 6 | 275.03x | 5 | 5 |

### Problem A - ciag akcji (plan z heurystyka)
1. `move_c3_s3_s4`
2. `move_c4_s5_s3`
3. `move_c2_s2_s5`
4. `move_c1_s1_s2`
5. `move_c5_s6_s1`
6. `move_c4_s3_s6`

### Problem B - ciag akcji (plan z heurystyka)
1. `move_c5_s6_s4`
2. `move_c3_s3_s6`
3. `move_c4_s5_s3`
4. `move_c3_s6_s5`

### Problem C - ciag akcji (plan z heurystyka)
1. `move_c4_s5_s4`
2. `move_c5_s6_s5`
3. `move_c2_s2_s6`
4. `move_c3_s3_s2`
5. `move_c2_s6_s3`
