# Example: How to Use Fuzz Testing Markers

This document provides practical examples of how to use the fuzz testing markers for implementing property-based testing and fuzzing in the PyTorch distributed test suite.

## Example 1: Finding All Fuzz Testing Candidates

### Using grep
```bash
cd test/distributed
grep -r "FUZZ_TESTING_CANDIDATE" . --include="*.py" | wc -l
# Output: 180
```

### Using the helper script
```bash
python find_fuzz_candidates.py --stats
```

Output:
```
================================================================================
FUZZ TESTING CANDIDATES STATISTICS
================================================================================
Total candidates: 180
  - With parametrized tests: 59
  - With random data generation: 174
  - With both parametrized and random data: 53
  - With hypothesis: 0
...
```

## Example 2: Identifying High-Priority Test Files

Find test files with both parametrized tests AND random data generation (highest priority for fuzz testing):

```bash
python find_fuzz_candidates.py --both
```

Output shows 53 high-priority files like:
- `test_c10d_common.py`
- `test_c10d_nccl.py`
- `fsdp/test_fsdp_core.py`
- etc.

## Example 3: Implementing Fuzz Testing with Hypothesis

### Before (original test with random data):

```python
# test_example.py
import torch

def test_tensor_operation():
    # Fixed size random tensor
    x = torch.rand(10, 10)
    result = my_distributed_op(x)
    assert result.shape == (10, 10)
```

### After (with hypothesis fuzz testing):

```python
# test_example.py
# FUZZ_TESTING_CANDIDATE: This test uses hypothesis and random data generation
import torch
from hypothesis import given, settings
import hypothesis.strategies as st

@given(
    size=st.integers(min_value=1, max_value=100),
    dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16])
)
@settings(max_examples=100, deadline=None)
def test_tensor_operation_fuzz(size, dtype):
    """Fuzz test with varying tensor sizes and dtypes."""
    x = torch.rand(size, size, dtype=dtype)
    result = my_distributed_op(x)
    
    # Property-based assertions
    assert result.shape == (size, size), "Shape should be preserved"
    assert result.dtype == dtype, "Dtype should be preserved"
    assert not torch.isnan(result).any(), "No NaN values"
    assert torch.isfinite(result).all(), "All values should be finite"
```

## Example 4: Expanding Parametrized Tests

### Before (limited parameter coverage):

```python
from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests

class TestDistributedOp(TestCase):
    @parametrize("backend", ["nccl", "gloo"])
    @parametrize("world_size", [2, 4])
    def test_collective(self, backend, world_size):
        ...
```

### After (expanded for fuzz testing):

```python
# FUZZ_TESTING_CANDIDATE: This test uses parametrized tests
from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests

class TestDistributedOp(TestCase):
    @parametrize("backend", ["nccl", "gloo", "mpi"])
    @parametrize("world_size", [1, 2, 3, 4, 8, 16])  # Added edge cases
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
    @parametrize("size", [1, 100, 1000, 10000])  # Various sizes
    def test_collective(self, backend, world_size, dtype, size):
        ...
```

## Example 5: CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Fuzz Testing
on:
  schedule:
    - cron: '0 2 * * *'  # Run nightly
  workflow_dispatch:  # Allow manual trigger

jobs:
  fuzz-test-high-priority:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install torch hypothesis pytest pytest-randomly
      
      - name: Find high-priority fuzz candidates
        run: |
          cd test/distributed
          python find_fuzz_candidates.py --both --export-json fuzz_list.json
          echo "High priority files:"
          cat fuzz_list.json | jq -r '.[].path'
      
      - name: Run fuzz tests with increased examples
        run: |
          cd test/distributed
          # Run each high-priority file with hypothesis
          export HYPOTHESIS_MAX_EXAMPLES=1000
          for file in $(cat fuzz_list.json | jq -r '.[].path'); do
            echo "Fuzzing $file"
            pytest "$file" -v --hypothesis-show-statistics || true
          done
```

## Example 6: Using pytest-randomly

Add randomization to test execution order to catch order-dependent bugs:

```bash
# Install pytest-randomly
pip install pytest-randomly

# Run tests with random seed
pytest test/distributed/test_c10d_common.py --randomly-seed=auto -v

# Run with specific seed for reproducibility
pytest test/distributed/test_c10d_common.py --randomly-seed=12345 -v
```

## Example 7: Custom Fuzz Testing Script

Create a script to run all high-priority candidates with extended fuzzing:

```python
#!/usr/bin/env python3
# fuzz_runner.py

import subprocess
import json
from pathlib import Path

def run_fuzz_tests():
    # Get high-priority candidates
    result = subprocess.run(
        ["python", "find_fuzz_candidates.py", "--both", "--export-json", "/tmp/fuzz.json"],
        cwd="test/distributed",
        capture_output=True
    )
    
    with open("/tmp/fuzz.json") as f:
        candidates = json.load(f)
    
    print(f"Running fuzz tests on {len(candidates)} high-priority files...")
    
    for candidate in candidates:
        filepath = candidate['absolute_path']
        print(f"\n{'='*80}")
        print(f"Fuzzing: {candidate['path']}")
        print(f"{'='*80}")
        
        # Run with hypothesis settings
        env = {
            "HYPOTHESIS_MAX_EXAMPLES": "500",
            "HYPOTHESIS_DERANDOMIZE": "1",  # Deterministic for CI
        }
        
        result = subprocess.run(
            ["pytest", filepath, "-v", "--hypothesis-show-statistics"],
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ FAILED: {candidate['path']}")
            print(result.stdout)
            print(result.stderr)
        else:
            print(f"✅ PASSED: {candidate['path']}")

if __name__ == "__main__":
    run_fuzz_tests()
```

## Example 8: Analyzing Test Coverage

Find which categories have the most fuzz testing opportunities:

```bash
python find_fuzz_candidates.py --stats
```

Output shows:
```
Breakdown by category:
  - _composable: 24
  - checkpoint: 29
  - fsdp: 35
  - tensor: 35
  ...
```

This helps prioritize which areas need more fuzz testing coverage.

## Example 9: Generating Test Reports

Export to JSON and analyze with jq:

```bash
# Export all candidates
python find_fuzz_candidates.py --export-json all_candidates.json

# Find files with only random data (no parametrize)
jq '[.[] | select((.reasons | contains(["random_data"])) and (.reasons | contains(["parametrized"]) | not))]' all_candidates.json | jq length

# Find FSDP-related tests
jq '[.[] | select(.path | contains("fsdp"))]' all_candidates.json | jq -r '.[].path'

# Count by reasons
jq '[.[] | .reasons[]] | group_by(.) | map({reason: .[0], count: length})' all_candidates.json
```

## Example 10: Property-Based Testing Pattern

A complete example of property-based testing for distributed operations:

```python
# FUZZ_TESTING_CANDIDATE: This test uses hypothesis and random data generation
import torch
import torch.distributed as dist
from hypothesis import given, settings, assume
import hypothesis.strategies as st

class TestDistributedFuzz(TestCase):
    
    @given(
        tensor_size=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=4),
        dtype=st.sampled_from([torch.float32, torch.float16, torch.int64]),
        reduce_op=st.sampled_from([dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN])
    )
    @settings(max_examples=200, deadline=None)
    def test_allreduce_properties(self, tensor_size, dtype, reduce_op):
        """Property-based test for allreduce operation."""
        
        # Assume constraints
        assume(all(s > 0 for s in tensor_size))
        
        # Create tensor
        tensor = torch.rand(*tensor_size, dtype=dtype)
        original = tensor.clone()
        
        # Perform operation
        dist.all_reduce(tensor, op=reduce_op)
        
        # Property checks
        assert tensor.shape == original.shape, "Shape should be preserved"
        assert tensor.dtype == original.dtype, "Dtype should be preserved"
        
        if reduce_op == dist.ReduceOp.SUM:
            # For SUM, result should be original * world_size
            expected = original * dist.get_world_size()
            assert torch.allclose(tensor, expected), "SUM property violated"
        
        # Idempotency for MAX/MIN
        if reduce_op in [dist.ReduceOp.MAX, dist.ReduceOp.MIN]:
            tensor2 = tensor.clone()
            dist.all_reduce(tensor2, op=reduce_op)
            assert torch.equal(tensor, tensor2), "Idempotency property violated"
```

## Best Practices

1. **Start Small**: Begin with high-priority files (both parametrized and random data)
2. **Use Seeds**: Always use deterministic seeds in CI for reproducibility
3. **Set Deadlines**: Use `deadline=None` in hypothesis for slow distributed tests
4. **Document Properties**: Clearly document what properties you're testing
5. **Monitor Coverage**: Use `--hypothesis-show-statistics` to see coverage
6. **Integrate Gradually**: Add fuzz testing incrementally, don't try to convert everything at once
7. **Keep Original Tests**: Maintain existing tests alongside fuzz tests
8. **Use Markers**: Add pytest markers for easy test selection (e.g., `@pytest.mark.fuzz`)

## Resources

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [pytest-randomly](https://github.com/pytest-dev/pytest-randomly)
- [Property-Based Testing](https://increment.com/testing/in-praise-of-property-based-testing/)
- PyTorch Distributed Testing: `torch.testing._internal.common_distributed`
