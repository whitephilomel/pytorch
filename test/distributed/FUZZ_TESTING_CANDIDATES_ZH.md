# test/distributed 中的模糊测试候选文件

本文档标识了 `test/distributed` 目录中适合进行模糊测试的测试文件。

## 概述

我们分析了 `test/distributed` 目录中的所有测试文件，以确定哪些文件适合进行模糊测试。如果文件满足以下一个或多个标准，则将其标记为模糊测试候选文件：

1. **使用参数化测试**：使用 `@parametrize`、`instantiate_parametrized_tests` 或类似机制的文件
2. **生成随机数据**：使用 `torch.rand`、`torch.randn`、`torch.randint` 或 Python 的 `random` 模块生成随机测试数据的文件
3. **使用基于属性的测试**：使用 hypothesis 或类似基于属性的测试框架的文件

## 统计信息

- **测试文件总数**：241
- **模糊测试候选文件**：180 (74.7%)
- **非候选文件**：61 (25.3%)

## 标记的文件

标记为模糊测试候选文件的文件在文件顶部有一个特殊注释：
```python
# FUZZ_TESTING_CANDIDATE: This test uses [parametrized tests|random data generation|hypothesis] and is suitable for fuzz testing
```

## 类别

### 使用参数化测试的文件（适合模糊测试）

这些文件使用参数化测试装饰器，可以从扩展的参数范围或随机参数生成中受益：

- 使用 `@parametrize` 装饰器的文件
- 使用 `instantiate_parametrized_tests` 的文件
- 使用 `@given`（hypothesis）的文件

### 使用随机数据生成的文件（适合模糊测试）

这些文件已经生成随机测试数据，可以从以下方面受益：
- 更激进的随机化
- 基于种子的可重现性
- 边缘情况生成
- 基于属性的测试

常见模式：
- `torch.rand()` / `torch.randn()` / `torch.randint()`
- `random.randint()` / `random.choice()` / `random.random()`
- `np.random.*`

## 完整的模糊测试候选文件列表

### _composable/fsdp/
- `test_fully_shard_autograd.py` - 随机数据生成
- `test_fully_shard_clip_grad_norm_.py` - 随机数据生成
- `test_fully_shard_comm.py` - 随机数据生成
- `test_fully_shard_compile.py` - 随机数据生成
- `test_fully_shard_extensions.py` - 随机数据生成
- `test_fully_shard_frozen.py` - 随机数据生成
- `test_fully_shard_grad_scaler.py` - 随机数据生成
- `test_fully_shard_ignore_params.py` - 参数化测试，随机数据
- `test_fully_shard_init.py` - 随机数据生成
- `test_fully_shard_logging.py` - 随机数据生成
- `test_fully_shard_memory.py` - 随机数据生成
- `test_fully_shard_mixed_precision.py` - 随机数据生成
- `test_fully_shard_overlap.py` - 随机数据生成
- `test_fully_shard_state.py` - 随机数据生成
- `test_fully_shard_state_dict.py` - 随机数据生成
- `test_fully_shard_training.py` - 随机数据生成

### _composable/test_composability/
- `test_2d_composability.py` - 参数化测试，随机数据
- `test_pp_composability.py` - 参数化测试，随机数据

### _composable/
- `test_checkpoint.py` - 随机数据生成
- `test_contract.py` - 随机数据生成
- `test_replicate.py` - 随机数据生成
- `test_replicate_mixed_precision.py` - 随机数据生成
- `test_replicate_training.py` - 随机数据生成
- `test_replicate_with_compiler.py` - 随机数据生成
- `test_replicate_with_fsdp.py` - 随机数据生成

### _shard/
- `sharded_optim/test_sharded_optim.py` - 随机数据生成
- `sharded_tensor/ops/test_embedding.py` - 随机数据生成
- `sharded_tensor/ops/test_embedding_bag.py` - 随机数据生成
- `sharded_tensor/test_sharded_tensor.py` - 随机数据生成
- `sharded_tensor/test_sharded_tensor_reshard.py` - 随机数据生成
- `sharding_plan/test_sharding_plan.py` - 随机数据生成
- `sharding_spec/test_sharding_spec.py` - 随机数据生成

### _tools/
- `test_fake_collectives.py` - 随机数据生成
- `test_fsdp2_mem_tracker.py` - 随机数据生成
- `test_mem_tracker.py` - 随机数据生成
- `test_memory_tracker.py` - 随机数据生成
- `test_mod_tracker.py` - 随机数据生成
- `test_runtime_estimator.py` - 随机数据生成
- `test_sac_estimator.py` - 随机数据生成
- `test_sac_ilp.py` - 随机数据生成

### algorithms/ddp_comm_hooks/
- `test_ddp_hooks.py` - 随机数据生成

### checkpoint/
大量检查点相关测试文件，都使用参数化测试和/或随机数据生成

### fsdp/
大量 FSDP（全分片数据并行）相关测试文件

### tensor/
大量分布式张量相关测试文件

### 根测试文件
- `test_c10d_common.py` - 参数化测试，随机数据
- `test_c10d_nccl.py` - 参数化测试，随机数据
- `test_collective_utils.py` - 参数化测试，随机数据
- `test_composability.py` - 参数化测试，随机数据
- 以及更多...

（完整列表见英文版文档）

## 模糊测试建议

### 高优先级候选文件

同时具有参数化测试和随机数据生成的文件是最高优先级：

1. `test_c10d_common.py` - 核心分布式通信原语
2. `test_c10d_nccl.py` - NCCL 集合操作
3. `test_collective_utils.py` - 集合实用函数
4. `test_composability.py` - 可组合性测试
5. FSDP 测试 - 全分片数据并行测试
6. 检查点测试 - 状态字典和检查点
7. 张量并行测试 - 张量并行

### 中等优先级候选文件

只有随机数据生成或只有参数化测试的文件仍然可以从模糊测试中受益，但可能需要更多设置。

### 实施策略

1. **对于参数化测试**：扩展参数范围，添加边缘情况，使用基于属性的测试
2. **对于随机数据测试**：使用基于种子的可重现性，添加 hypothesis 装饰器，使用各种分布进行测试
3. **添加模糊测试基础设施**：考虑使用 hypothesis、pytest-randomly 或自定义模糊测试框架

## 非候选文件

没有参数化测试或随机数据生成的文件（61个文件）通常是：
- 配置或实用程序模块
- 仅具有固定测试用例的测试
- 具有特定场景的集成测试
- 需要特定硬件或环境设置的测试

这些文件仍然可以进行测试，但可能不会像传统的模糊测试方法那样受益。

## 如何使用此标记

1. 搜索所有带有 `FUZZ_TESTING_CANDIDATE` 标记的文件
2. 查看文件以了解测试的性质
3. 根据您的模糊测试目标选择合适的候选文件
4. 实施模糊测试策略（hypothesis、参数扩展等）
5. 集成到 CI/CD 管道中以持续进行模糊测试

## 查找命令

```bash
# 查找所有模糊测试候选文件
grep -r "FUZZ_TESTING_CANDIDATE" test/distributed --include="*.py"

# 统计候选文件数量
grep -r "FUZZ_TESTING_CANDIDATE" test/distributed --include="*.py" | wc -l

# 查找使用参数化测试的文件
grep -r "FUZZ_TESTING_CANDIDATE.*parametrized" test/distributed --include="*.py"

# 查找使用随机数据的文件
grep -r "FUZZ_TESTING_CANDIDATE.*random data" test/distributed --include="*.py"
```
