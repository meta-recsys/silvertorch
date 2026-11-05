# TPU/JAX Support

SilverTorch's TPU path is implemented with JAX/XLA. The existing PyTorch custom
operators use C++/CUDA kernels and do not lower to TPU.

## Files

- `silvertorch/tpu/`: experimental JAX implementations.
- `silvertorch/tpu/tests/`: CPU-backed JAX unit tests.
- `examples/tpu_bloom_smoke.py`: small smoke test intended to run on a TPU JAX
  image.

## CPU Test

Use a Python environment with JAX installed, then run:

```bash
python -m unittest silvertorch.tpu.tests.test_bloom -v
```

If JAX is not installed in the base environment, install the TPU extra or use
your own JAX environment:

```bash
python -m pip install -e '.[tpu]'
python -m unittest silvertorch.tpu.tests.test_bloom -v
```

## TPU Smoke Test

Run the smoke-test example in an environment where JAX can see TPU devices:

```bash
python examples/tpu_bloom_smoke.py
```

The example checks `jax.default_backend()` and fails unless the backend is
`tpu`.
