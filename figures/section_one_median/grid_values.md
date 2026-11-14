# Representative long-run return grids

Empirical discounted returns from each traversable start, rendered with the repository notebook grid visualizer. These are policy evaluation values, not a learned state-value function.

## History-4 GRU

| Row / column | 0 | 1 | 2 | 3 |
|:--|--:|--:|:--|--:|
| 0 | 0.563 | 0.530 | c · 0.000 | 0.800 |
| 1 | 0.558 | 0.548 | a · 0.553 | b/T · 1.000 |
| 2 | B | 0.541 | c · 0.000 | 0.800 |
| 3 | b/T · 1.000 | 0.790 | a/T · 1.000 | 0.810 |
| 4 | 0.980 | c · 0.000 | 0.770 | c · 0.000 |

## Current-reading MLP

| Row / column | 0 | 1 | 2 | 3 |
|:--|--:|--:|:--|--:|
| 0 | 0.000 | 0.000 | c · 0.000 | 0.000 |
| 1 | 0.000 | 0.020 | a · 0.670 | b/T · 1.000 |
| 2 | B | 0.020 | c · 0.000 | 0.010 |
| 3 | b/T · 1.000 | 0.050 | a/T · 0.940 | 0.650 |
| 4 | 0.000 | c · 0.000 | 0.000 | c · 0.000 |

## GRU minus MLP

| Row / column | 0 | 1 | 2 | 3 |
|:--|--:|--:|:--|--:|
| 0 | 0.563 | 0.530 | c · 0.000 | 0.800 |
| 1 | 0.558 | 0.528 | a · -0.117 | b/T · 0.000 |
| 2 | B | 0.521 | c · 0.000 | 0.790 |
| 3 | b/T · 0.000 | 0.740 | a/T · 0.060 | 0.160 |
| 4 | 0.980 | c · 0.000 | 0.770 | c · 0.000 |
