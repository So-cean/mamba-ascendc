# Benchmarks

三个算子 benchmark 共用同一组 `[B,L,H,P,N,C,G]` case 和 FP32 输入：

```bash
# NVIDIA A100，官方 mamba_ssm
python benchmarks/mamba2_gpu_bench.py --cases medium extreme --warmup 30 --repeat 200

# Ascend NPU，Triton-Ascend
python benchmarks/mamba2_triton_ascend_bench.py --cases medium extreme --warmup 30 --repeat 200

# Ascend NPU，AscendC
python benchmarks/mamba2_npu_final_bench.py --cases medium extreme --warmup 30 --repeat 200 --skip-precision
```

每个 case 输出一行 JSON。公平比较要求：

1. 三端使用相同 case、FP32、warmup 和 repeat。
2. 计时前完成输入迁移和 kernel 编译。
3. 使用设备 Event，只统计 SSD forward。
4. 分别记录设备型号、软件版本和是否启用了 fallback。
5. 精度由独立测试门禁保证，不在性能循环中执行 reference。

`vmamba2_network_bench.py` 用于纯 Vision-Mamba2 网络测试，可分别选择
`--device cuda`、`--device npu` 或 `--device cpu`。网络结果不能与算子级结果混用。
