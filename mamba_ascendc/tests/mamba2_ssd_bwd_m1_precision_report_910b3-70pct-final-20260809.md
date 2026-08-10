# Mamba2 SSD backward M1 精度验证报告

- 测试时间：2026-08-09T16:20:02
- 平台：Ascend910B3
- 参考：PyTorch FP32 `ssd_chunk_scan_ref` autograd
- M1：AscendC off/diag-state/state-passing/dt native-core 集成路径
- 通过条件：每项梯度 NRMSE <= 8e-3 且 cosine >= 0.999

## 总览

| 总用例 | 通过 | 失败 | Worst NRMSE | Lowest cosine |
|---:|---:|---:|---:|---:|
| 41 | 41 | 0 | 1.078e-03 | 0.999992847 |

## 用例结果

| # | 类别 | Shape `[B,L,H,G]` | 模式 | Worst gradient | NRMSE | Lowest cosine | 结果 |
|---:|---|---|---|---|---:|---:|---|
| 1 | single | `[1, 64, 1, 1]` | output_only | B | 5.861e-05 | 0.999999881 | PASS |
| 2 | single | `[1, 64, 1, 1]` | output_final | B | 6.410e-05 | 0.999999940 | PASS |
| 3 | single | `[1, 64, 1, 1]` | final_only | B | 1.867e-04 | 0.999999881 | PASS |
| 4 | single | `[1, 64, 1, 1]` | full_channel_D | dt | 4.749e-04 | 0.999999881 | PASS |
| 5 | single | `[1, 64, 1, 1]` | full_head_D | dt | 5.351e-04 | 0.999999881 | PASS |
| 6 | heads | `[1, 64, 2, 1]` | output_only | x | 5.548e-05 | 0.999999762 | PASS |
| 7 | heads | `[1, 64, 2, 1]` | output_final | B | 5.322e-05 | 0.999999940 | PASS |
| 8 | heads | `[1, 64, 2, 1]` | final_only | B | 1.084e-04 | 1.000000000 | PASS |
| 9 | heads | `[1, 64, 2, 1]` | full_channel_D | dt | 6.458e-04 | 0.999999285 | PASS |
| 10 | heads | `[1, 64, 2, 1]` | full_head_D | dt | 2.931e-04 | 0.999999762 | PASS |
| 11 | groups | `[1, 128, 2, 2]` | output_only | B | 1.063e-04 | 0.999998927 | PASS |
| 12 | groups | `[1, 128, 2, 2]` | output_final | B | 1.149e-04 | 0.999999106 | PASS |
| 13 | groups | `[1, 128, 2, 2]` | final_only | B | 1.852e-04 | 0.999999940 | PASS |
| 14 | groups | `[1, 128, 2, 2]` | full_channel_D | dt | 6.352e-04 | 0.999999642 | PASS |
| 15 | groups | `[1, 128, 2, 2]` | full_head_D | dt | 5.124e-04 | 0.999999583 | PASS |
| 16 | shared | `[1, 128, 4, 1]` | output_only | B | 5.771e-05 | 0.999999881 | PASS |
| 17 | shared | `[1, 128, 4, 1]` | output_final | B | 6.276e-05 | 0.999998152 | PASS |
| 18 | shared | `[1, 128, 4, 1]` | final_only | B | 9.704e-05 | 0.999999523 | PASS |
| 19 | shared | `[1, 128, 4, 1]` | full_channel_D | dt | 6.644e-04 | 0.999999762 | PASS |
| 20 | shared | `[1, 128, 4, 1]` | full_head_D | dt | 5.120e-04 | 0.999999046 | PASS |
| 21 | multi_group | `[1, 128, 4, 2]` | output_only | B | 8.178e-05 | 1.000000000 | PASS |
| 22 | multi_group | `[1, 128, 4, 2]` | output_final | B | 8.282e-05 | 0.999999762 | PASS |
| 23 | multi_group | `[1, 128, 4, 2]` | final_only | B | 1.255e-04 | 0.999999285 | PASS |
| 24 | multi_group | `[1, 128, 4, 2]` | full_channel_D | dt | 6.600e-04 | 0.999999762 | PASS |
| 25 | multi_group | `[1, 128, 4, 2]` | full_head_D | dt | 5.514e-04 | 0.999999702 | PASS |
| 26 | batch | `[2, 64, 4, 4]` | output_only | B | 5.761e-05 | 0.999999285 | PASS |
| 27 | batch | `[2, 64, 4, 4]` | output_final | B | 6.365e-05 | 0.999999404 | PASS |
| 28 | batch | `[2, 64, 4, 4]` | final_only | B | 1.837e-04 | 1.000000000 | PASS |
| 29 | batch | `[2, 64, 4, 4]` | full_channel_D | dt | 4.805e-04 | 0.999999523 | PASS |
| 30 | batch | `[2, 64, 4, 4]` | full_head_D | dt | 3.957e-04 | 0.999999464 | PASS |
| 31 | chunks | `[1, 256, 8, 2]` | output_only | B | 6.311e-05 | 0.999999642 | PASS |
| 32 | chunks | `[1, 256, 8, 2]` | output_final | B | 7.273e-05 | 0.999998987 | PASS |
| 33 | chunks | `[1, 256, 8, 2]` | final_only | dt | 1.139e-04 | 0.999999404 | PASS |
| 34 | chunks | `[1, 256, 8, 2]` | full_channel_D | dt | 5.773e-04 | 0.999999583 | PASS |
| 35 | chunks | `[1, 256, 8, 2]` | full_head_D | dt | 7.365e-04 | 0.999999523 | PASS |
| 36 | occupancy | `[2, 128, 8, 8]` | output_only | B | 1.050e-04 | 0.999999166 | PASS |
| 37 | occupancy | `[2, 128, 8, 8]` | output_final | B | 1.073e-04 | 0.999999464 | PASS |
| 38 | occupancy | `[2, 128, 8, 8]` | final_only | B | 1.890e-04 | 0.999998391 | PASS |
| 39 | occupancy | `[2, 128, 8, 8]` | full_channel_D | dt | 5.091e-04 | 0.999998271 | PASS |
| 40 | occupancy | `[2, 128, 8, 8]` | full_head_D | dt | 5.620e-04 | 0.999998152 | PASS |
| 41 | hybrid_threshold | `[1, 1024, 64, 16]` | full_head_D | dt | 1.078e-03 | 0.999992847 | PASS |

## 覆盖范围

- output-only、output+final、final-only 三种上游梯度入口。
- D 的 `[H,P]` 与 `[H]` 两种接口，以及 z、dt_bias、softplus、finite clamp、initial state。
- B/L/H/G scaling 和共享 group 的确定性 head reduction。
- 九项 public 梯度按实际 optional 输入逐项比较。

该报告验证 M1 native-core 数学与 public autograd 闭环；SSD 核心分支均已原生化，public 层仍保留少量 tensor glue。
