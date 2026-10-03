# Every frame corrupted: POD performance by artifact type

### Variant: fixed (every frame corrupted; POD ensemble 400 pairs, auto rank A=8, B=8; PIV on 150 pairs)

| Condition | RMS error (px) | Median (px) | Vectors > 1 px | Artifact induced RMS (px) |
|---|---|---|---|---|
| clean | 0.080 | 0.032 | 0.02% | 0.000 |
| dirty | 0.669 | 0.079 | 4.03% | 0.665 |
| pod | 0.090 | 0.043 | 0.02% | 0.058 |
| highpass | 0.187 | 0.050 | 0.15% | 0.177 |
| pod+highpass | 0.091 | 0.046 | 0.02% | 0.064 |

### Variant: drift (every frame corrupted; POD ensemble 400 pairs, auto rank A=17, B=17; PIV on 150 pairs)

| Condition | RMS error (px) | Median (px) | Vectors > 1 px | Artifact induced RMS (px) |
|---|---|---|---|---|
| clean | 0.081 | 0.032 | 0.02% | 0.000 |
| dirty | 0.890 | 0.085 | 5.88% | 0.887 |
| pod | 0.093 | 0.046 | 0.03% | 0.059 |
| highpass | 0.132 | 0.050 | 0.07% | 0.119 |
| pod+highpass | 0.094 | 0.048 | 0.03% | 0.066 |

### Variant: random (every frame corrupted; POD ensemble 400 pairs, auto rank A=57, B=57; PIV on 150 pairs)

| Condition | RMS error (px) | Median (px) | Vectors > 1 px | Artifact induced RMS (px) |
|---|---|---|---|---|
| clean | 0.078 | 0.031 | 0.01% | 0.000 |
| dirty | 1.205 | 0.086 | 8.11% | 1.203 |
| pod | 0.102 | 0.058 | 0.01% | 0.079 |
| highpass | 0.303 | 0.051 | 0.25% | 0.297 |
| pod+highpass | 0.103 | 0.058 | 0.02% | 0.082 |

### Variant: random_streaks (every frame corrupted; POD ensemble 400 pairs, auto rank A=8, B=8; PIV on 150 pairs)

| Condition | RMS error (px) | Median (px) | Vectors > 1 px | Artifact induced RMS (px) |
|---|---|---|---|---|
| clean | 0.077 | 0.032 | 0.01% | 0.000 |
| dirty | 0.979 | 0.111 | 7.87% | 0.975 |
| pod | 0.165 | 0.053 | 0.18% | 0.151 |
| highpass | 0.250 | 0.060 | 0.31% | 0.242 |
| pod+highpass | 0.131 | 0.052 | 0.06% | 0.114 |

