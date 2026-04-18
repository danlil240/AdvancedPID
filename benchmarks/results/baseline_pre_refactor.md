# Benchmark summary — baseline_pre_refactor

- total cases: **18**
- ok: **18**
- errors: **0**
- stable fraction: **77.78%**
- median IAE: **1.130**
- median overshoot: **13.02 %**
- median settling time: **4.040 s**

| case | status | Kp | Ki | Kd | IAE | overshoot% | settling_s | stable |
|------|--------|----|----|----|-----|-----------|------------|--------|
| `fopdt_K0.5_tau5.0_th0.5` | ok | 57.047 | 16.975 | 11.982 | 5.18 | 10.6 | 19.99 | False |
| `fopdt_K0.5_tau2.0_th0.5` | ok | 7.260 | 4.987 | 1.821 | 1.13 | 36.4 | 4.01 | True |
| `fopdt_K0.5_tau1.0_th0.5` | ok | 3.592 | 2.393 | 0.704 | 0.93 | 12.0 | 3.03 | True |
| `fopdt_K0.5_tau1.0_th1.0` | ok | 2.226 | 1.199 | 0.855 | 1.79 | 9.8 | 5.48 | True |
| `fopdt_K1.0_tau5.0_th0.5` | ok | 11.823 | 8.184 | 3.607 | 1.21 | 47.8 | 3.82 | True |
| `fopdt_K1.0_tau2.0_th0.5` | ok | 3.575 | 2.395 | 0.873 | 1.12 | 34.6 | 4.05 | True |
| `fopdt_K1.0_tau1.0_th0.5` | ok | 1.769 | 1.243 | 0.342 | 0.92 | 12.8 | 2.98 | True |
| `fopdt_K1.0_tau1.0_th1.0` | ok | 1.107 | 0.598 | 0.412 | 1.80 | 9.7 | 5.52 | True |
| `fopdt_K2.5_tau5.0_th0.5` | ok | 3.997 | 3.283 | 1.274 | 1.35 | 68.8 | 3.41 | True |
| `fopdt_K2.5_tau2.0_th0.5` | ok | 1.461 | 0.958 | 0.365 | 1.11 | 35.1 | 4.03 | True |
| `fopdt_K2.5_tau1.0_th0.5` | ok | 0.722 | 0.497 | 0.144 | 0.92 | 13.2 | 2.95 | True |
| `fopdt_K2.5_tau1.0_th1.0` | ok | 0.444 | 0.240 | 0.172 | 1.79 | 9.5 | 5.48 | True |
| `first_K1.0_tau1.0` | ok | 1.200 | 1.198 | 0.038 | 0.82 | 0.0 | 3.17 | True |
| `first_K2.0_tau3.0` | ok | 178.742 | 50.051 | 0.604 | 1.13 | 4.9 | 19.99 | False |
| `first_K0.5_tau0.5` | ok | 1.200 | 2.400 | 0.019 | 0.82 | 0.0 | 3.21 | True |
| `second_wn1_zeta0.3` | ok | 126.472 | 88.431 | 0.512 | 5.03 | 61.8 | 19.99 | False |
| `second_wn1_zeta0.7` | ok | 126.009 | 88.431 | 0.510 | 2.63 | 41.8 | 19.99 | False |
| `second_wn1_zeta1.5` | ok | 124.975 | 88.431 | 0.506 | 0.70 | 19.9 | 6.96 | True |
