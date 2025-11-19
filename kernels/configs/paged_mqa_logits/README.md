## Compiled Info
- [`triton`](https://github.com/ROCm/triton/tree/gluon_ext) (3.5.0+gite392a058)
- [`aiter`](https://github.com/ROCm/aiter/commit/e2a1a6f7c8628e14b28c09844ee25ef0b6f9b19d) (e2a1a6f)

Generate Ahead-of-time compiled kernel
```bash
# generate paged_mqa_logits_preshuffle_64x256x128_B64PFW2
python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py --kv_preshuffle --blocksize=64 -aot
# generate paged_mqa_logits_preshuffle_64x256x128_B16PFW2
python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py --kv_preshuffle --blocksize=16 -aot
# generate paged_mqa_logits_64x256x128_B1PTW2
python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py -p -aot
```

## How to use these kernels

refer to `https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/pa_mqa_logits.py#L16-L21`

