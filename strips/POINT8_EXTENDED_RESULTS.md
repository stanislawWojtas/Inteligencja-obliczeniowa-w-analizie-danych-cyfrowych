# Punkt 8 - Dodatkowe problemy z podcelami

## Definicje dodatkowych problemow

Wszystkie problemy startuja ze stanu `('c1','c2','c3','X','c4','c5')` i maja po 2 podcele.

### Problem D
- Podcel 1: `('c2', 'c1', 'c4', 'c5', 'c3', 'X')`
- Podcel 2: `('c1', 'c2', 'c3', 'c4', 'c5', 'X')`
- Cel koncowy: `('c2', 'c1', 'c4', 'c3', 'X', 'c5')`

### Problem E
- Podcel 1: `('c2', 'c1', 'c4', 'X', 'c5', 'c3')`
- Podcel 2: `('c1', 'c2', 'c3', 'c5', 'X', 'c4')`
- Cel koncowy: `('c2', 'c1', 'c4', 'X', 'c5', 'c3')`

### Problem F
- Podcel 1: `('c2', 'c1', 'c5', 'c4', 'X', 'c3')`
- Podcel 2: `('c1', 'c2', 'c4', 'c3', 'X', 'c5')`
- Cel koncowy: `('c2', 'c1', 'c3', 'c4', 'c5', 'X')`

## Wyniki

Uzyto: `Forward_STRIPS` + `SearcherMPP`, etapowo: start -> podcel1 -> podcel2 -> cel.

| Problem | Plan bez h | Plan z h | Czas bez h [s] | Czas z h [s] | Rozszerzone bez h | Rozszerzone z h | Dlugosc bez h | Dlugosc z h | Segmenty z h | Warunek >=20 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| D | tak | tak | 0.337536 | 0.003187 | 2103 | 75 | 21 | 21 | 7 + 7 + 7 | tak |
| E | tak | tak | 0.337454 | 0.003274 | 2124 | 75 | 21 | 21 | 7 + 7 + 7 | tak |
| F | tak | tak | 0.337802 | 0.003271 | 2125 | 76 | 21 | 21 | 7 + 7 + 7 | tak |

### Problem D - ciag akcji (plan z podcelami i heurystyka)
1. `move_c2_s2_s4`
2. `move_c1_s1_s2`
3. `move_c3_s3_s1`
4. `move_c4_s5_s3`
5. `move_c3_s1_s5`
6. `move_c2_s4_s1`
7. `move_c5_s6_s4`
8. `move_c4_s3_s6`
9. `move_c3_s5_s3`
10. `move_c5_s4_s5`
11. `move_c1_s2_s4`
12. `move_c2_s1_s2`
13. `move_c1_s4_s1`
14. `move_c4_s6_s4`
15. `move_c4_s4_s6`
16. `move_c3_s3_s4`
17. `move_c1_s1_s3`
18. `move_c2_s2_s1`
19. `move_c1_s3_s2`
20. `move_c4_s6_s3`
21. `move_c5_s5_s6`

### Problem E - ciag akcji (plan z podcelami i heurystyka)
1. `move_c2_s2_s4`
2. `move_c1_s1_s2`
3. `move_c5_s6_s1`
4. `move_c3_s3_s6`
5. `move_c4_s5_s3`
6. `move_c5_s1_s5`
7. `move_c2_s4_s1`
8. `move_c4_s3_s4`
9. `move_c3_s6_s3`
10. `move_c1_s2_s6`
11. `move_c2_s1_s2`
12. `move_c1_s6_s1`
13. `move_c4_s4_s6`
14. `move_c5_s5_s4`
15. `move_c2_s2_s5`
16. `move_c1_s1_s2`
17. `move_c3_s3_s1`
18. `move_c4_s6_s3`
19. `move_c3_s1_s6`
20. `move_c2_s5_s1`
21. `move_c5_s4_s5`

### Problem F - ciag akcji (plan z podcelami i heurystyka)
1. `move_c2_s2_s4`
2. `move_c1_s1_s2`
3. `move_c5_s6_s1`
4. `move_c3_s3_s6`
5. `move_c5_s1_s3`
6. `move_c2_s4_s1`
7. `move_c4_s5_s4`
8. `move_c2_s1_s5`
9. `move_c1_s2_s1`
10. `move_c3_s6_s2`
11. `move_c5_s3_s6`
12. `move_c4_s4_s3`
13. `move_c3_s2_s4`
14. `move_c2_s5_s2`
15. `move_c4_s3_s5`
16. `move_c3_s4_s3`
17. `move_c1_s1_s4`
18. `move_c2_s2_s1`
19. `move_c1_s4_s2`
20. `move_c4_s5_s4`
21. `move_c5_s6_s5`
