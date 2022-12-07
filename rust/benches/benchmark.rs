use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use itertools::Itertools;
use pprof::criterion::{Output, PProfProfiler};

extern crate nn_backend_test;
use crate::nn_backend_test::nn_runners::{CompiledNNRunner, Runner, TractOnnxRunner};

#[cfg(tflitec)]
use crate::nn_backend_test::nn_runners::TfLiteRunner;

const CLASSIFIER_PATH: &str = "../models/hulks_2022/classifier.hdf5";
const CLASSIFIER_PATH_ONNX: &str = "../models/hulks_2022/classifier.onnx";
const CLASSIFIER_PATH_TFLITE: &str = "../models/hulks_2022/classifier.tflite";

const BALL_SAMPLE_PATH: &str = "../data/ball_sample.png";

fn criterion_benchmark(c: &mut Criterion) {
    // open image, resize it and make a Tensor out of it
    let image = image::open(BALL_SAMPLE_PATH).unwrap().to_rgb8();

    // extract red channel only
    let input_buffer = image.pixels().map(|f| f[0] as f32).collect_vec();

    // options used by different backends
    let input_shape = [32, 32, 1];
    let thread_count = 1;

    let mut group = c.benchmark_group("NN Runner");

    group.bench_with_input(
        BenchmarkId::new("CompiledNNRunner", "ball"),
        &input_buffer,
        |b, input_buffer| {
            b.iter_batched_ref(
                || -> CompiledNNRunner { CompiledNNRunner::new(CLASSIFIER_PATH).unwrap() },
                |runner| runner.run_inference_single_io(input_buffer).len(),
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::new("TractOnnxRunner", "ball"),
        &input_buffer,
        |b, input_buffer| {
            b.iter_batched_ref(
                || -> TractOnnxRunner {
                    TractOnnxRunner::new(CLASSIFIER_PATH_ONNX, &input_shape).unwrap()
                },
                |runner| runner.run_inference_single_io(input_buffer).len(),
                BatchSize::SmallInput,
            )
        },
    );

    cfg_if::cfg_if! {
        if #[cfg(not(nao))] {
            group.bench_with_input(
                BenchmarkId::new("TfLiteRunner", BALL_SAMPLE_PATH),
                &input_buffer,
                |b, input_buffer| {
                    b.iter_batched_ref(
                        || -> TfLiteRunner {
                            TfLiteRunner::new(CLASSIFIER_PATH_TFLITE, thread_count).unwrap()
                        },
                        |runner| runner.run_inference_single_io(input_buffer).len(),
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_benchmark
}

criterion_main!(benches);
