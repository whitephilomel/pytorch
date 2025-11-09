# Fuzz Testing Guide for test/distributed

[English](#english) | [中文](#中文)

---

## English

### Overview

This directory contains test files marked for fuzz testing. Files marked with `FUZZ_TESTING_CANDIDATE` are suitable for fuzz testing based on their use of:

- **Parametrized tests** (`@parametrize`, `instantiate_parametrized_tests`)
- **Random data generation** (`torch.rand`, `torch.randn`, `torch.randint`, `random`)
- **Hypothesis property-based testing**

### Quick Start

#### Find all fuzz testing candidates:
```bash
grep -r "FUZZ_TESTING_CANDIDATE" . --include="*.py"
```

#### Use the helper script:
```bash
# Show statistics
python find_fuzz_candidates.py --stats

# List all candidates
python find_fuzz_candidates.py --list

# Show high-priority candidates (both parametrized and random data)
python find_fuzz_candidates.py --both

# Show only parametrized test files
python find_fuzz_candidates.py --parametrized

# Show only random data generation files
python find_fuzz_candidates.py --random-data

# Export to JSON for automated processing
python find_fuzz_candidates.py --export-json fuzz_candidates.json
```

### Statistics

- **Total test files**: 241
- **Fuzz testing candidates**: 180 (74.7%)
- **Files with parametrized tests**: 59
- **Files with random data generation**: 174
- **High-priority files (both criteria)**: 53

### Marker Format

Each marked file contains a comment like:
```python
# FUZZ_TESTING_CANDIDATE: This test uses parametrized tests and random data generation
```

This marker appears near the top of the file, after the owner comment.

### Implementing Fuzz Testing

#### 1. For Parametrized Tests

Expand parameter ranges to include edge cases:

```python
# Before
@parametrize("size", [10, 100])
def test_function(size):
    ...

# After (with fuzz testing)
@parametrize("size", [0, 1, 10, 100, 1000, 2**16-1])
def test_function(size):
    ...
```

#### 2. For Random Data Tests

Add hypothesis for property-based testing:

```python
from hypothesis import given
import hypothesis.strategies as st

# Before
def test_function():
    x = torch.rand(10, 10)
    result = my_function(x)
    assert result.shape == (10, 10)

# After (with hypothesis)
@given(st.integers(min_value=1, max_value=100))
def test_function(size):
    x = torch.rand(size, size)
    result = my_function(x)
    assert result.shape == (size, size)
```

#### 3. Using pytest-randomly

Add randomization to test order:

```bash
pip install pytest-randomly
pytest --randomly-seed=auto test_file.py
```

### High-Priority Candidates

Files with **both** parametrized tests and random data generation:

1. **Core distributed communication**:
   - `test_c10d_common.py`
   - `test_c10d_nccl.py`
   - `test_collective_utils.py`

2. **FSDP (Fully Sharded Data Parallel)**:
   - `fsdp/test_fsdp_comm.py`
   - `fsdp/test_fsdp_core.py`
   - `fsdp/test_fsdp_checkpoint.py`

3. **Checkpointing**:
   - `checkpoint/test_file_system_checkpoint.py`
   - `checkpoint/test_fsdp_optim_state.py`
   - `checkpoint/test_hsdp_checkpoint.py`

4. **Tensor Parallel**:
   - `tensor/test_dtensor_compile.py`
   - `tensor/test_matrix_ops.py`
   - `tensor/parallel/test_micro_pipeline_tp.py`

### Integration with CI/CD

#### Using in GitHub Actions:

```yaml
name: Fuzz Testing
on: [push, pull_request]

jobs:
  fuzz-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Find fuzz candidates
        run: |
          cd test/distributed
          python find_fuzz_candidates.py --both --export-json fuzz_list.json
      - name: Run fuzz tests
        run: |
          # Run tests with increased randomization
          pytest --randomly-seed=auto $(cat fuzz_list.json | jq -r '.[].absolute_path')
```

### Documentation Files

- **FUZZ_TESTING_CANDIDATES.md** - Detailed list of all candidates (English)
- **FUZZ_TESTING_CANDIDATES_ZH.md** - Detailed list of all candidates (Chinese)
- **find_fuzz_candidates.py** - Helper script to find and analyze candidates

### Contributing

When adding new tests to `test/distributed`:

1. If your test uses parametrized tests or random data generation, add the marker:
   ```python
   # FUZZ_TESTING_CANDIDATE: This test uses [your criteria]
   ```

2. Run the helper script to verify:
   ```bash
   python find_fuzz_candidates.py --stats
   ```

3. Consider implementing property-based testing with hypothesis for better coverage

---

## 中文

### 概述

此目录包含标记用于模糊测试的测试文件。带有 `FUZZ_TESTING_CANDIDATE` 标记的文件适合进行模糊测试，基于它们使用：

- **参数化测试** (`@parametrize`, `instantiate_parametrized_tests`)
- **随机数据生成** (`torch.rand`, `torch.randn`, `torch.randint`, `random`)
- **Hypothesis 基于属性的测试**

### 快速开始

#### 查找所有模糊测试候选文件：
```bash
grep -r "FUZZ_TESTING_CANDIDATE" . --include="*.py"
```

#### 使用辅助脚本：
```bash
# 显示统计信息
python find_fuzz_candidates.py --stats

# 列出所有候选文件
python find_fuzz_candidates.py --list

# 显示高优先级候选文件（同时具有参数化和随机数据）
python find_fuzz_candidates.py --both

# 仅显示参数化测试文件
python find_fuzz_candidates.py --parametrized

# 仅显示随机数据生成文件
python find_fuzz_candidates.py --random-data

# 导出为 JSON 用于自动处理
python find_fuzz_candidates.py --export-json fuzz_candidates.json
```

### 统计信息

- **测试文件总数**：241
- **模糊测试候选文件**：180 (74.7%)
- **带参数化测试的文件**：59
- **带随机数据生成的文件**：174
- **高优先级文件（两个标准都满足）**：53

### 标记格式

每个标记的文件包含如下注释：
```python
# FUZZ_TESTING_CANDIDATE: This test uses parametrized tests and random data generation
```

此标记出现在文件顶部附近，在所有者注释之后。

### 实施模糊测试

#### 1. 对于参数化测试

扩展参数范围以包含边缘情况：

```python
# 之前
@parametrize("size", [10, 100])
def test_function(size):
    ...

# 之后（带模糊测试）
@parametrize("size", [0, 1, 10, 100, 1000, 2**16-1])
def test_function(size):
    ...
```

#### 2. 对于随机数据测试

添加 hypothesis 进行基于属性的测试：

```python
from hypothesis import given
import hypothesis.strategies as st

# 之前
def test_function():
    x = torch.rand(10, 10)
    result = my_function(x)
    assert result.shape == (10, 10)

# 之后（使用 hypothesis）
@given(st.integers(min_value=1, max_value=100))
def test_function(size):
    x = torch.rand(size, size)
    result = my_function(x)
    assert result.shape == (size, size)
```

#### 3. 使用 pytest-randomly

为测试顺序添加随机化：

```bash
pip install pytest-randomly
pytest --randomly-seed=auto test_file.py
```

### 高优先级候选文件

**同时**具有参数化测试和随机数据生成的文件：

1. **核心分布式通信**：
   - `test_c10d_common.py`
   - `test_c10d_nccl.py`
   - `test_collective_utils.py`

2. **FSDP（全分片数据并行）**：
   - `fsdp/test_fsdp_comm.py`
   - `fsdp/test_fsdp_core.py`
   - `fsdp/test_fsdp_checkpoint.py`

3. **检查点**：
   - `checkpoint/test_file_system_checkpoint.py`
   - `checkpoint/test_fsdp_optim_state.py`
   - `checkpoint/test_hsdp_checkpoint.py`

4. **张量并行**：
   - `tensor/test_dtensor_compile.py`
   - `tensor/test_matrix_ops.py`
   - `tensor/parallel/test_micro_pipeline_tp.py`

### 文档文件

- **FUZZ_TESTING_CANDIDATES.md** - 所有候选文件的详细列表（英文）
- **FUZZ_TESTING_CANDIDATES_ZH.md** - 所有候选文件的详细列表（中文）
- **find_fuzz_candidates.py** - 用于查找和分析候选文件的辅助脚本

### 贡献

在向 `test/distributed` 添加新测试时：

1. 如果您的测试使用参数化测试或随机数据生成，请添加标记：
   ```python
   # FUZZ_TESTING_CANDIDATE: This test uses [your criteria]
   ```

2. 运行辅助脚本以验证：
   ```bash
   python find_fuzz_candidates.py --stats
   ```

3. 考虑使用 hypothesis 实现基于属性的测试以获得更好的覆盖率
