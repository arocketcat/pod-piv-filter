# POD artifact filter benchmark

1000 image pairs (2000 images), 225000 vectors per condition. POD modes removed: A = 10, B = 10.

## All vectors

| Condition | RMS error (px) | Median (px) | 95th pct (px) | > 0.5 px | > 1 px | Flagged | Artifact induced RMS (px) |
|---|---|---|---|---|---|---|---|
| Clean (no artifacts) | 0.0795 | 0.0317 | 0.1459 | 0.33% | 0.02% | 0.89% | 0.0000 |
| Corrupted, unfiltered | 2.0000 | 0.0601 | 0.4489 | 4.30% | 1.91% | 2.69% | 1.9991 |
| Corrupted, min subtraction | 1.3890 | 0.0504 | 0.3009 | 2.16% | 0.88% | 1.72% | 1.3875 |
| Corrupted, POD filtered | 0.0875 | 0.0393 | 0.1625 | 0.37% | 0.02% | 0.85% | 0.0514 |

## windows on fixed streaks (88000 vectors)

| Condition | RMS error (px) | Median (px) | 95th pct (px) | > 0.5 px | > 1 px | Flagged | Artifact induced RMS (px) |
|---|---|---|---|---|---|---|---|
| Clean (no artifacts) | 0.0762 | 0.0362 | 0.1425 | 0.23% | 0.00% | 0.49% | 0.0000 |
| Corrupted, unfiltered | 2.7328 | 0.1248 | 0.7079 | 8.20% | 3.45% | 4.38% | 2.7326 |
| Corrupted, min subtraction | 1.9287 | 0.0880 | 0.4170 | 3.70% | 1.51% | 2.24% | 1.9279 |
| Corrupted, POD filtered | 0.0850 | 0.0443 | 0.1605 | 0.27% | 0.00% | 0.48% | 0.0497 |

## hotspot windows when flashing (47610 vectors)

| Condition | RMS error (px) | Median (px) | 95th pct (px) | > 0.5 px | > 1 px | Flagged | Artifact induced RMS (px) |
|---|---|---|---|---|---|---|---|
| Clean (no artifacts) | 0.0688 | 0.0367 | 0.1388 | 0.04% | 0.00% | 0.02% | 0.0000 |
| Corrupted, unfiltered | 3.0028 | 0.1203 | 1.0937 | 10.43% | 5.28% | 5.06% | 3.0025 |
| Corrupted, min subtraction | 1.9256 | 0.0956 | 0.5793 | 5.83% | 2.92% | 3.09% | 1.9251 |
| Corrupted, POD filtered | 0.0804 | 0.0450 | 0.1574 | 0.10% | 0.00% | 0.06% | 0.0486 |

## pairs with global flash (36000 vectors)

| Condition | RMS error (px) | Median (px) | 95th pct (px) | > 0.5 px | > 1 px | Flagged | Artifact induced RMS (px) |
|---|---|---|---|---|---|---|---|
| Clean (no artifacts) | 0.0798 | 0.0317 | 0.1452 | 0.32% | 0.02% | 0.90% | 0.0000 |
| Corrupted, unfiltered | 1.8971 | 0.0598 | 0.4439 | 4.17% | 1.87% | 2.70% | 1.8953 |
| Corrupted, min subtraction | 1.5836 | 0.0502 | 0.3069 | 2.36% | 1.11% | 1.88% | 1.5823 |
| Corrupted, POD filtered | 0.0890 | 0.0413 | 0.1632 | 0.37% | 0.03% | 0.86% | 0.0530 |

## pairs with random streak (15075 vectors)

| Condition | RMS error (px) | Median (px) | 95th pct (px) | > 0.5 px | > 1 px | Flagged | Artifact induced RMS (px) |
|---|---|---|---|---|---|---|---|
| Clean (no artifacts) | 0.0769 | 0.0318 | 0.1425 | 0.26% | 0.02% | 0.81% | 0.0000 |
| Corrupted, unfiltered | 1.4900 | 0.0629 | 0.4124 | 3.60% | 1.41% | 2.34% | 1.4878 |
| Corrupted, min subtraction | 1.1395 | 0.0532 | 0.2802 | 1.83% | 0.72% | 1.54% | 1.1375 |
| Corrupted, POD filtered | 0.0874 | 0.0412 | 0.1655 | 0.31% | 0.02% | 0.78% | 0.0547 |

## artifact free windows quiet pairs (25942 vectors)

| Condition | RMS error (px) | Median (px) | 95th pct (px) | > 0.5 px | > 1 px | Flagged | Artifact induced RMS (px) |
|---|---|---|---|---|---|---|---|
| Clean (no artifacts) | 0.0860 | 0.0279 | 0.1554 | 0.54% | 0.03% | 1.55% | 0.0000 |
| Corrupted, unfiltered | 0.2437 | 0.0321 | 0.1699 | 0.88% | 0.24% | 1.57% | 0.2314 |
| Corrupted, min subtraction | 0.0911 | 0.0306 | 0.1628 | 0.61% | 0.04% | 1.53% | 0.0526 |
| Corrupted, POD filtered | 0.0938 | 0.0344 | 0.1673 | 0.57% | 0.05% | 1.41% | 0.0513 |

## Mode sweep (100 pairs)

| Modes removed | RMS error (px) | > 0.5 px | Artifact induced RMS (px) |
|---|---|---|---|
| 0 | 1.6017 | 4.38% | 1.6003 |
| 1 | 2.5264 | 2.96% | 2.5249 |
| 2 | 1.7692 | 1.40% | 1.7679 |
| 3 | 0.0963 | 0.43% | 0.0673 |
| 4 | 0.0886 | 0.36% | 0.0541 |
| 5 | 0.0881 | 0.36% | 0.0532 |
| 6 | 0.0877 | 0.37% | 0.0530 |
| 7 | 0.0871 | 0.34% | 0.0503 |
| 8 | 0.0873 | 0.39% | 0.0521 |
| 9 | 0.0880 | 0.36% | 0.0512 |
| 10 | 0.0872 | 0.35% | 0.0511 |
| 11 | 0.0872 | 0.35% | 0.0517 |
| 12 | 0.0878 | 0.36% | 0.0516 |
| 14 | 0.0879 | 0.36% | 0.0518 |
| 17 | 0.0882 | 0.36% | 0.0530 |
| 20 | 0.0900 | 0.40% | 0.0537 |
| 30 | 0.0905 | 0.38% | 0.0575 |
| 50 | 0.0923 | 0.36% | 0.0605 |
| 80 | 0.0954 | 0.32% | 0.0684 |

Same pairs, reference RMS: clean 0.0798, dirty 1.6017, minsub 1.4306, pod 0.0872
