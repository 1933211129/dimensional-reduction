# Ours vs MCB-AR Direct Diagnostic Study

All values are reported as mean ± std over the three public datasets and two folds under each missingness protocol.

| Protocol | Method | Blocks | Omitted pairs | Omission rate | Selected attrs | RR (%) | F1 | RL | Mean reduct Jaccard with Ours |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MCAR | Ours | 20.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 99.435 $\pm$ 0.315 | 0.269 $\pm$ 0.180 | 0.441 $\pm$ 0.113 | 1.000 $\pm$ 0.000 |
| MCAR | MCB-AR | 20.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 99.435 $\pm$ 0.315 | 0.278 $\pm$ 0.209 | 0.454 $\pm$ 0.138 | 0.000 $\pm$ 0.000 |
| Object-wise | Ours | 18.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 99.435 $\pm$ 0.315 | 0.257 $\pm$ 0.193 | 0.449 $\pm$ 0.138 | 1.000 $\pm$ 0.000 |
| Object-wise | MCB-AR | 1.000 $\pm$ 0.000 | 169.167 $\pm$ 13.790 | 1.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 100.000 $\pm$ 0.000 | 0.286 $\pm$ 0.227 | 0.476 $\pm$ 0.165 | 0.000 $\pm$ 0.000 |
| Attribute-wise | Ours | 20.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 99.435 $\pm$ 0.315 | 0.334 $\pm$ 0.200 | 0.458 $\pm$ 0.123 | 1.000 $\pm$ 0.000 |
| Attribute-wise | MCB-AR | 20.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 99.435 $\pm$ 0.315 | 0.334 $\pm$ 0.200 | 0.458 $\pm$ 0.123 | 1.000 $\pm$ 0.000 |
| Blockwise | Ours | 20.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 99.435 $\pm$ 0.315 | 0.334 $\pm$ 0.200 | 0.458 $\pm$ 0.123 | 1.000 $\pm$ 0.000 |
| Blockwise | MCB-AR | 20.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 0.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 99.435 $\pm$ 0.315 | 0.334 $\pm$ 0.200 | 0.458 $\pm$ 0.123 | 1.000 $\pm$ 0.000 |
