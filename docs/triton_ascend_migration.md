# Triton-Ascend (910B/A2) 迁移笔记

本文记录将 GPU Triton 算子迁移到昇腾 910B (Ascend A2) 时的关键注意事项、踩坑记录与最佳实践。

---

## 1. 核心原则

- **放弃 GPU「逻辑 grid 自由定义」**，转为昇腾「物理核组绑定」。
- **grid 优先用 1D**，2D 写法在 NPU 上也会合并为 1D，实际效果等价。
- **coreDim ≤ 65535**，超过会触发 `coreDim=xxxx can't be greater than UINT16_MAX`。
- **VV 场景要求 32 字节访存对齐**，CV 场景要求 512 字节对齐。
- **移除所有 GPU 专属 API**：`num_warps`、`num_stages`、`__syncthreads` 等。

---

## 2. 编译器限制与陷阱

### 2.1 `range()` 运行时循环 → 慎用或避免嵌套

**问题**：`range()` 运行时循环（`scf.for`）在 BiShengIR 中代码生成存在缺陷。当外层 `range()` 嵌套内层 `static_range()` 时，特定条件下（grid 较大或 `num_blocks ≥ 2`）编译器会错误处理跨迭代变量依赖，导致计算结果错乱。

**表现**：
- `num_blocks=1` 时结果正确
- `num_blocks≥2` 时出现巨大数值差异（如 `diff=1.75` 或 `diff=3.3e11`）

**解决**：
- 外层循环改用 `tl.static_range(MAX_BLOCKS)` + `if block_id < num_blocks:`
- 但 `if` 在 `static_range` 内可能导致 `scf.if op expects a non-empty block` 错误（见 2.6）
- **最佳实践**：尽量避免需要跨块 carry 的循环结构，改用 `tl.associative_scan` 做块内 scan

### 2.2 `tl.zeros()` → 编译器崩溃

**问题**：`tl.zeros([BLOCK_N], dtype=tl.float32)` 在 BiShengIR 编译阶段触发 `double free or corruption` 或 `SIGABRT`。

**解决**：使用 `tl.full([BLOCK_N], 0.0, dtype=tl.float32)` 替代。

### 2.3 `tl.full([...])` 常量索引不支持

**问题**：`tensor[BLOCK_L - 1]` 形式的常量索引在 Triton-Ascend 上不被支持。

**表现**：`ValueError('unsupported tensor index: constexpr[3]')`

**影响**：无法从向量化数组中提取单个元素（如获取 carry）。

**解决**：
- 避免需要从向量中提取单个元素的设计
- 如需 carry，在 host 侧处理或使用标量变量而非数组索引

### 2.4 `tl.atomic_add` → 性能灾难

**问题**：`tl.atomic_add` 在 Triton-Ascend 上性能极差，对于中等规模配置（如 2×128×16×256）单次 backward 执行时间超过 **10 分钟**。

**解决**：
- **彻底消除 `atomic_add`**：改用 grid 扩展策略（如 grid=(batch×dim×dstate,)）+ partial buffer
- 每个 program 写入独立的 partial buffer，host 侧用 `torch.sum` 聚合
- 此策略将 backward 时间从 10 分钟降至 ~20ms

### 2.5 `tl.arange()` 参数必须是 `tl.constexpr`

**问题**：`tl.arange(0, K)` 中的 `K` 必须是 `tl.constexpr`，运行时参数会报错。

**表现**：`ValueError("arange's arguments must be of type tl.constexpr")`

**解决**：将相关维度（如 `dstate`、`BLOCK_L`）在 host 侧计算后以关键字参数传入：
```python
kernel[grid](..., BV=BV, K=K, BLOCK_L=BLOCK_L)
```

### 2.6 `tl.load` 的 mask 类型

**问题**：`tl.load(..., mask=mask)` 中的 mask 必须是 `tl.int1`（bool），不能是 `int32`。

**表现**：`ValueError('Unsupported ptr type <[64, 4], int32> in tl.load')`

**解决**：显式转换：
```python
mask = (offsets < seqlen).to(tl.int1)
```

### 2.7 `if` 语句在 `static_range` 内 → `scf.if` 空 block 错误

**问题**：`for n in tl.static_range(MAX_DSTATE): if n < dstate: ...` 在 BiShengIR 中触发 `scf.if op expects a non-empty block`。

**解决**：
- 移除 `if` 条件，确保 `MAX_DSTATE == dstate`（即编译时确定的循环次数等于实际维度）
- 或者将条件逻辑改写为 `tl.where` 掩码操作

---

## 3. `tl.associative_scan` 使用指南

### 3.1 基本用法

`tl.associative_scan` 在 Ascend A2/A3 上**可用**，支持 `tuple of Tensor`，无需像 GPU 那样将两个 float32 打包成 uint64。

**combine_fn 签名**：
```python
@triton.jit
def combine_fn(x1, a1, x2, a2):
    return (a2 * x1 + x2, a2 * a1)
```
注意：**输入 tuple 会被拆开为独立参数传入**，不是 `(left, right)` 两个 tuple。

### 3.2 数据对齐要求

文档要求：
> 不使用 mask 过滤掉多余数据索引

这意味着 `associative_scan` 的输入必须是完整对齐的 block。但可以通过 `tl.load` 的 `other` 参数实现 padding：

```python
# padding 位置: x=0, a=1 (exp(0)=1)
x_vec = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
a_vec = tl.load(a_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
h_vec, a_cum = tl.associative_scan((x_vec, a_vec), axis=0, combine_fn=combine_fn)
```

### 3.3 不支持 `reverse=True`

Ascend `associative_scan` 不支持 `reverse=True`，反向 scan 需要手动实现（或 host 侧反转序列）。

### 3.4 数值精度差异

`associative_scan` 使用 tree-based parallel scan，与顺序累加的浮点精度存在差异：
- 绝对误差可能较大（大数值时可达 1~2）
- **相对误差通常 < 1%**，在 float32 正常范围内
- 建议测试时使用**相对误差**而非绝对误差：
```python
rel_diff = ((out_ref - out_tri).abs() / (out_ref.abs() + 1e-6)).max()
assert rel_diff < 1e-2  # 1% 相对误差
```

---

## 4. 内存与性能约束

### 4.1 UB (Unified Buffer) 容量

- **910B UB = 192KB**
- 所有向量化 load 的中间结果必须 fit 在 UB 内
- 建议估算峰值内存使用，公式：
```
peak_ub = num_vectors * vector_size * dtype_size
```

### 4.2 coreDim 限制

- `grid_dim = batch * dim * ...` 不能超过 **65535**
- 若超过，可设置环境变量 `TRITON_ALL_BLOCKS_PARALLEL=1`
- 或增大 BLOCK_SIZE 减少 grid 大小

### 4.3 编译时间

- `static_range` 完全展开会显著增加编译时间
- `BLOCK_L=32` 编译约 15~30s
- `BLOCK_L=128` 可能超时（10min+）
- 建议 `BLOCK_L ≤ 64` 作为经验上限

---

## 5. 推荐架构设计

基于以上约束，推荐的 selective scan forward kernel 设计：

```
Grid: (batch, dim)  或  1D (batch*dim,)
每个 program: 处理一个 (b, d)
内部:
  - 加载 delta[BLOCK_L], u[BLOCK_L]（共享）
  - static_range(dstate) 循环每个 n
  - 每次循环:
    - 加载 B[n, BLOCK_L], C[n, BLOCK_L]
    - a = exp(delta * A_n)
    - x = delta * B * u
    - h, _ = associative_scan((x, a), axis=0, combine_fn=...)
    - y_acc += h * C
  - 存储 y_acc[BLOCK_L]
```

**优势**：
- 无 `atomic_add`
- 无 `range()` 嵌套 `static_range()`
- 利用 `associative_scan` 做正确且高效的 scan
- `dstate` 用 `static_range` 避免 `scf.if` 问题

---

## 6. 环境变量速查

| 变量 | 作用 |
|------|------|
| `TRITON_ALL_BLOCKS_PARALLEL=1` | 自动根据物理核数优化逻辑核数，解决 coreDim 超限 |
| `TRITON_DEBUG=1` | 保存 IR 文件到 `~/.triton/cache/` 用于调试 |
| `TRITON_KERNEL_OVERRIDE=1` | 启用 kernel override（调试） |
| `TRITON_KERNEL_DUMP=1` | 启用 IR dump（调试） |

---

## 7. 常见错误速查

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| `coreDim=xxxx can't be greater than UINT16_MAX` | grid 超过 65535 | 增大 BLOCK_SIZE 或设置 `TRITON_ALL_BLOCKS_PARALLEL=1` |
| `ub overflow, requires xxxx bits while 1572684 bits available!` | UB 内存超限 | 减小 BLOCK_SIZE 或分块处理 |
| `arange's arguments must be of type tl.constexpr` | `tl.arange` 参数不是 constexpr | 将维度改为 `tl.constexpr` 参数传入 |
| `Unsupported ptr type <[..., int32>` | `tl.load` mask 类型错误 | mask 用 `.to(tl.int1)` 转换 |
| `double free or corruption` | `tl.zeros()` 触发编译器 bug | 改用 `tl.full(..., 0.0, ...)` |
| `scf.if op expects a non-empty block` | `static_range` 内有 `if` 语句 | 移除 `if` 或改用 `tl.where` |
| `unsupported tensor index: constexpr[x]` | 对向量做常量索引 | 避免 `vec[constant_idx]`，改用标量变量 |

---

## 8. 参考资源

- [Triton-Ascend 官方文档](https://triton-ascend-test.readthedocs.io/zh-cn/latest/)
- [Triton-Ascend 迁移指南](https://triton-ascend-test.readthedocs.io/zh-cn/latest/migration_guide/migrate_from_gpu.html)
- [associative_scan API](https://triton-ascend-test.readthedocs.io/zh-cn/latest/triton_api/Scan_Sort_Ops/associative_scan.html)
- 原始 GPU 实现参考：[sustcsonglin/mamba-triton](https://github.com/sustcsonglin/mamba-triton)
