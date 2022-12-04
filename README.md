# Evaluating runtime performance of Neural Network inference backends

## Rust

```bash
cd rust/

# Build
cargo build

# Benchmark
cargo bench
```
Example output for benchmarking:
```bash
NN Runner/CompiledNNRunner/../data/ball_sample.png
                        time:   [47.885 µs 48.143 µs 48.418 µs]
                        change: [-7.9907% -5.5533% -3.4340%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 5 outliers among 100 measurements (5.00%)
  5 (5.00%) high mild
NN Runner/TractOnnxRunner/../data/ball_sample.png
                        time:   [174.39 µs 175.50 µs 176.72 µs]
                        change: [-4.7379% -2.8932% -1.1718%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 5 outliers among 100 measurements (5.00%)
  4 (4.00%) high mild
  1 (1.00%) high severe
NN Runner/TfLiteRunner/../data/ball_sample.png
                        time:   [270.93 µs 274.30 µs 278.63 µs]
                        change: [-0.8163% +0.6843% +2.4751%] (p = 0.42 > 0.05)
                        No change in performance detected.
Found 6 outliers among 100 measurements (6.00%)
  3 (3.00%) high mild
  3 (3.00%) high severe
  
```

## C++

WIP

## Cross compiling

With Yocto or other CMake, Cargo friendly methods, this should work out of the box in general.

```
source ___ # yocto environment setup

# Follow same for Rust
cargo build
```

For a yocto toolchain with a `x86_64-aldebaran-linux-gnu` identifier, results are found in `rust/target/x86_64-aldebaran-linux-gnu`.

Tested with toolchain of [HULKs](https://github.com/hulks/hulk). `tflitec` dependency doesn't work due to its dependency in bazel build (unless we supply libtensorflow_lite_c.so)
