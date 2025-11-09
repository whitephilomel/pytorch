# Fuzz Testing Candidates in test/distributed

This document identifies test files in `test/distributed` that are good candidates for fuzz testing.

## Overview

We have analyzed all test files in the `test/distributed` directory to identify which ones would benefit from fuzz testing. A file is marked as a fuzz testing candidate if it meets one or more of the following criteria:

1. **Uses Parametrized Tests**: Files that use `@parametrize`, `instantiate_parametrized_tests`, or similar mechanisms
2. **Generates Random Data**: Files that generate random test data using `torch.rand`, `torch.randn`, `torch.randint`, or Python's `random` module
3. **Uses Property-Based Testing**: Files that use hypothesis or similar property-based testing frameworks

## Statistics

- **Total test files**: 241
- **Fuzz testing candidates**: 180 (74.7%)
- **Non-candidates**: 61 (25.3%)

## Marked Files

Files marked as fuzz testing candidates have a special comment at the top of the file:
```python
# FUZZ_TESTING_CANDIDATE: This test uses [parametrized tests|random data generation|hypothesis] and is suitable for fuzz testing
```

## Categories

### Files with Parametrized Tests (Good for Fuzz Testing)

These files use parametrized test decorators and can benefit from expanded parameter ranges or random parameter generation:

- Files using `@parametrize` decorator
- Files using `instantiate_parametrized_tests`
- Files using `@given` (hypothesis)

### Files with Random Data Generation (Good for Fuzz Testing)

These files already generate random test data and can benefit from:
- More aggressive randomization
- Seed-based reproducibility
- Edge case generation
- Property-based testing

Common patterns:
- `torch.rand()` / `torch.randn()` / `torch.randint()`
- `random.randint()` / `random.choice()` / `random.random()`
- `np.random.*`

## Complete List of Fuzz Testing Candidates

### _composable/fsdp/
- `test_fully_shard_autograd.py` - Random data generation
- `test_fully_shard_clip_grad_norm_.py` - Random data generation
- `test_fully_shard_comm.py` - Random data generation
- `test_fully_shard_compile.py` - Random data generation
- `test_fully_shard_extensions.py` - Random data generation
- `test_fully_shard_frozen.py` - Random data generation
- `test_fully_shard_grad_scaler.py` - Random data generation
- `test_fully_shard_ignore_params.py` - Parametrized tests, Random data
- `test_fully_shard_init.py` - Random data generation
- `test_fully_shard_logging.py` - Random data generation
- `test_fully_shard_memory.py` - Random data generation
- `test_fully_shard_mixed_precision.py` - Random data generation
- `test_fully_shard_overlap.py` - Random data generation
- `test_fully_shard_state.py` - Random data generation
- `test_fully_shard_state_dict.py` - Random data generation
- `test_fully_shard_training.py` - Random data generation

### _composable/test_composability/
- `test_2d_composability.py` - Parametrized tests, Random data
- `test_pp_composability.py` - Parametrized tests, Random data

### _composable/
- `test_checkpoint.py` - Random data generation
- `test_contract.py` - Random data generation
- `test_replicate.py` - Random data generation
- `test_replicate_mixed_precision.py` - Random data generation
- `test_replicate_training.py` - Random data generation
- `test_replicate_with_compiler.py` - Random data generation
- `test_replicate_with_fsdp.py` - Random data generation

### _shard/
- `sharded_optim/test_sharded_optim.py` - Random data generation
- `sharded_tensor/ops/test_embedding.py` - Random data generation
- `sharded_tensor/ops/test_embedding_bag.py` - Random data generation
- `sharded_tensor/test_sharded_tensor.py` - Random data generation
- `sharded_tensor/test_sharded_tensor_reshard.py` - Random data generation
- `sharding_plan/test_sharding_plan.py` - Random data generation
- `sharding_spec/test_sharding_spec.py` - Random data generation

### _tools/
- `test_fake_collectives.py` - Random data generation
- `test_fsdp2_mem_tracker.py` - Random data generation
- `test_mem_tracker.py` - Random data generation
- `test_memory_tracker.py` - Random data generation
- `test_mod_tracker.py` - Random data generation
- `test_runtime_estimator.py` - Random data generation
- `test_sac_estimator.py` - Random data generation
- `test_sac_ilp.py` - Random data generation

### algorithms/ddp_comm_hooks/
- `test_ddp_hooks.py` - Random data generation

### checkpoint/
- `_experimental/test_checkpoint_process.py` - Random data generation
- `_experimental/test_checkpoint_reader.py` - Random data generation
- `_experimental/test_checkpointer.py` - Random data generation
- `_experimental/test_staging.py` - Random data generation
- `e2e/test_e2e_save_and_load.py` - Parametrized tests, Random data
- `e2e/test_fine_tuning.py` - Random data generation
- `fsdp/test_fsdp_dsd.py` - Random data generation
- `test_async_process_executor.py` - Random data generation
- `test_checkpoint.py` - Random data generation
- `test_dedup_tensors.py` - Random data generation
- `test_dtensor_resharding.py` - Parametrized tests, Random data
- `test_file_system_checkpoint.py` - Parametrized tests, Random data
- `test_file_system_checkpoint_cpu.py` - Parametrized tests, Random data
- `test_format_utils.py` - Random data generation
- `test_fsdp_model_state.py` - Random data generation
- `test_fsdp_optim_state.py` - Parametrized tests, Random data
- `test_fsdp_tp_checkpoint_conversion.py` - Random data generation
- `test_fsspec.py` - Random data generation
- `test_hf_safetensor_e2e.py` - Parametrized tests, Random data
- `test_hf_storage.py` - Random data generation
- `test_hsdp_checkpoint.py` - Parametrized tests, Random data
- `test_pg_transport.py` - Random data generation
- `test_planner.py` - Random data generation
- `test_save_load_api.py` - Random data generation
- `test_state_dict.py` - Random data generation
- `test_state_dict_stager.py` - Random data generation
- `test_state_dict_utils.py` - Random data generation
- `test_tp_checkpoint.py` - Random data generation
- `test_utils.py` - Random data generation

### fsdp/
- `test_checkpoint_wrapper.py` - Random data generation
- `test_distributed_checkpoint.py` - Parametrized tests
- `test_fsdp_backward_prefetch.py` - Random data generation
- `test_fsdp_checkpoint.py` - Parametrized tests, Random data
- `test_fsdp_clip_grad_norm.py` - Random data generation
- `test_fsdp_comm.py` - Parametrized tests, Random data
- `test_fsdp_comm_hooks.py` - Random data generation
- `test_fsdp_core.py` - Parametrized tests, Random data
- `test_fsdp_dtensor_state_dict.py` - Random data generation
- `test_fsdp_exec_order.py` - Random data generation
- `test_fsdp_fine_tune.py` - Random data generation
- `test_fsdp_flatten_params.py` - Random data generation
- `test_fsdp_fp16.py` - Random data generation
- `test_fsdp_freezing_weights.py` - Random data generation
- `test_fsdp_gradient_accumulation.py` - Random data generation
- `test_fsdp_hybrid_shard.py` - Parametrized tests, Random data
- `test_fsdp_ignored_modules.py` - Random data generation
- `test_fsdp_input.py` - Random data generation
- `test_fsdp_memory.py` - Random data generation
- `test_fsdp_meta.py` - Random data generation
- `test_fsdp_misc.py` - Parametrized tests, Random data
- `test_fsdp_mixed_precision.py` - Parametrized tests, Random data
- `test_fsdp_multiple_forward.py` - Random data generation
- `test_fsdp_multiple_wrapping.py` - Random data generation
- `test_fsdp_op_method_dispatch.py` - Random data generation
- `test_fsdp_optim_state.py` - Parametrized tests, Random data
- `test_fsdp_overlap.py` - Parametrized tests, Random data
- `test_fsdp_pure_fp16.py` - Random data generation
- `test_fsdp_set_state_dict.py` - Parametrized tests, Random data
- `test_fsdp_sharded_grad_scaler.py` - Random data generation
- `test_fsdp_state_dict.py` - Parametrized tests, Random data
- `test_fsdp_summon_full_params.py` - Random data generation
- `test_fsdp_tp_integration.py` - Random data generation
- `test_fsdp_traversal.py` - Random data generation
- `test_fsdp_uneven.py` - Random data generation
- `test_fsdp_unshard_params.py` - Random data generation
- `test_fsdp_use_orig_params.py` - Parametrized tests, Random data
- `test_hsdp_dtensor_state_dict.py` - Random data generation
- `test_shard_utils.py` - Random data generation
- `test_wrap.py` - Parametrized tests, Random data

### optim/
- `test_apply_optimizer_in_backward.py` - Random data generation
- `test_named_optimizer.py` - Random data generation
- `test_zero_redundancy_optimizer.py` - Parametrized tests, Random data

### pipelining/
- `test_backward.py` - Random data generation
- `test_microbatch.py` - Random data generation
- `test_pipe.py` - Parametrized tests, Random data
- `test_schedule.py` - Parametrized tests, Random data
- `test_schedule_multiproc.py` - Parametrized tests, Random data
- `test_stage.py` - Parametrized tests, Random data
- `test_transformer.py` - Random data generation
- `test_unflatten.py` - Random data generation

### tensor/debug/
- `test_comm_mode.py` - Random data generation
- `test_comm_mode_features.py` - Random data generation
- `test_debug_mode.py` - Parametrized tests, Random data
- `test_op_coverage.py` - Random data generation

### tensor/experimental/
- `test_local_map.py` - Random data generation
- `test_register_sharding.py` - Random data generation
- `test_tp_transform.py` - Random data generation

### tensor/parallel/
- `test_micro_pipeline_tp.py` - Parametrized tests, Random data
- `test_parallelize_api.py` - Random data generation
- `test_tp_examples.py` - Parametrized tests, Random data
- `test_tp_random_state.py` - Random data generation
- `test_tp_style.py` - Random data generation

### tensor/
- `test_api.py` - Random data generation
- `test_attention.py` - Random data generation
- `test_convolution_ops.py` - Random data generation
- `test_dtensor.py` - Random data generation
- `test_dtensor_compile.py` - Parametrized tests, Random data
- `test_dtensor_dispatch_overhead.py` - Random data generation
- `test_dtensor_export.py` - Parametrized tests, Random data
- `test_dtensor_ops.py` - Random data generation
- `test_dynamic.py` - Parametrized tests, Random data
- `test_embedding_ops.py` - Random data generation
- `test_experimental_ops.py` - Random data generation
- `test_init.py` - Random data generation
- `test_math_ops.py` - Random data generation
- `test_matrix_ops.py` - Parametrized tests, Random data
- `test_op_schema.py` - Random data generation
- `test_op_strategy.py` - Random data generation
- `test_pointwise_ops.py` - Random data generation
- `test_random_ops.py` - Random data generation
- `test_redistribute.py` - Parametrized tests, Random data
- `test_tensor_ops.py` - Random data generation
- `test_utils.py` - Random data generation
- `test_view_ops.py` - Random data generation
- `test_xla_integration.py` - Random data generation

### Root test files
- `test_aten_comm_compute_reordering.py` - Random data generation
- `test_c10d_common.py` - Parametrized tests, Random data
- `test_c10d_functional_native.py` - Random data generation
- `test_c10d_gloo.py` - Random data generation
- `test_c10d_nccl.py` - Parametrized tests, Random data
- `test_c10d_ops_nccl.py` - Random data generation
- `test_c10d_pypg.py` - Random data generation
- `test_c10d_spawn_gloo.py` - Random data generation
- `test_c10d_spawn_nccl.py` - Random data generation
- `test_c10d_ucc.py` - Random data generation
- `test_collective_utils.py` - Parametrized tests, Random data
- `test_composability.py` - Parametrized tests, Random data
- `test_cupy_as_tensor.py` - Random data generation
- `test_data_parallel.py` - Random data generation
- `test_device_mesh.py` - Random data generation
- `test_dynamo_distributed.py` - Random data generation
- `test_fake_pg.py` - Random data generation
- `test_functional_api.py` - Parametrized tests, Random data
- `test_inductor_collectives.py` - Parametrized tests, Random data
- `test_local_tensor.py` - Random data generation
- `test_multi_threaded_pg.py` - Random data generation
- `test_nccl.py` - Random data generation
- `test_nvshmem.py` - Parametrized tests, Random data
- `test_nvshmem_triton.py` - Parametrized tests
- `test_overlap_bucketing_unit.py` - Parametrized tests
- `test_p2p_ipc.py` - Random data generation
- `test_pg_wrapper.py` - Random data generation
- `test_serialization.py` - Random data generation
- `test_store.py` - Parametrized tests
- `test_symmetric_memory.py` - Parametrized tests, Random data

## Recommendations for Fuzz Testing

### High Priority Candidates

Files with both parametrized tests AND random data generation are the highest priority:

1. `test_c10d_common.py` - Core distributed communication primitives
2. `test_c10d_nccl.py` - NCCL collective operations
3. `test_collective_utils.py` - Collective utility functions
4. `test_composability.py` - Composability testing
5. FSDP tests - Fully Sharded Data Parallel testing
6. Checkpoint tests - State dict and checkpointing
7. Tensor parallel tests - Tensor parallelism

### Medium Priority Candidates

Files with only random data generation or only parametrized tests can still benefit from fuzz testing but may require more setup.

### Implementation Strategies

1. **For parametrized tests**: Expand parameter ranges, add edge cases, use property-based testing
2. **For random data tests**: Use seed-based reproducibility, add hypothesis decorators, test with various distributions
3. **Add fuzzing infrastructure**: Consider using hypothesis, pytest-randomly, or custom fuzzing frameworks

## Non-Candidates

Files without parametrized tests or random data generation (61 files) are typically:
- Configuration or utility modules
- Tests with fixed test cases only
- Integration tests with specific scenarios
- Tests that require specific hardware or environment setup

These can still be tested but may not benefit as much from traditional fuzz testing approaches.
