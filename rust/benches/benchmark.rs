use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use image::imageops;
use itertools::Itertools;
use pprof::criterion::{Output, PProfProfiler};

extern crate nn_backend_test;
use crate::nn_backend_test::nn_runners::{CompiledNNRunner, Runner, TractOnnxRunner};

#[cfg(feature = "tflite")]
use crate::nn_backend_test::nn_runners::TfLiteRunner;

// TODO find a configuration based approach to load all the possible cases and benchmark! these hardcoded paths are messy.


const CLASSIFIER_PATH: &str = "../models/hulks_2022/classifier_updated.hdf5";
const MULTI_CLASSIFIER_PATH: &str = "../models/hulks_2022/classifier_multiclass.hdf5";
const CLASSIFIER_PATH_ONNX: &str = "../models/hulks_2022/classifier.onnx";
const SEMANTIC_SEGMENTATION_DEEPLAB_ONNX: &str =
    "../models/semantic_segmentation/lite-model_deeplabv3_1_metadata_2_converted.onnx";

const _CLASSIFIER_PATH_TFLITE: &str = "../models/hulks_2022/classifier.tflite";
const _SEMANTIC_SEGMENTATION_DEEPLABL_TFLITE: &str =
    "../models/semantic_segmentation/lite-model_deeplabv3_1_metadata_2.tflite";
const mobilenet_: &str =
    "../models/semantic_segmentation/lite-model_deeplabv3_1_metadata_2.tflite";

const BALL_SAMPLE_PATH: &str = "../data/ball_sample.png";
const _SEMSEG_SAMPLE_PATH: &str =
    "../data/01024_GermanOpen2019_HULKs_Sabretooth-2nd_52240197_upper-002.png";

fn criterion_benchmark(c: &mut Criterion) {
    // open image, resize it and make a Tensor out of it
    let ball_image = image::open(BALL_SAMPLE_PATH).unwrap().to_rgb8();
    // extract red channel only
    let ball_input_buffer = ball_image.pixels().map(|f| f[0] as f32).collect_vec();

    // open image for semseg image, resize it and make a Tensor out of it
    let semseg_image = image::open(_SEMSEG_SAMPLE_PATH).unwrap();

    // options used by different backends
    let ball_model_input_shape = [32, 32, 1];
    let deeplab_input_shape = [1, 257, 257, 3];
    let _thread_count = 1;

    let mut group = c.benchmark_group("NN Runner");

    group.bench_with_input(
        BenchmarkId::new("CompiledNNRunner", "ball (single class)"),
        &ball_input_buffer,
        |b, ball_input_buffer| {
            b.iter_batched_ref(
                || -> CompiledNNRunner { CompiledNNRunner::new(CLASSIFIER_PATH).unwrap() },
                |runner| runner.run_inference_single_io(ball_input_buffer).len(),
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::new("CompiledNNRunner", "ball++ (multi class)"),
        &ball_input_buffer,
        |b, ball_input_buffer| {
            b.iter_batched_ref(
                || -> CompiledNNRunner { CompiledNNRunner::new(MULTI_CLASSIFIER_PATH).unwrap() },
                |runner| runner.run_inference_single_io(ball_input_buffer).len(),
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::new("TractOnnxRunner", "ball"),
        &ball_input_buffer,
        |b, ball_input_buffer| {
            b.iter_batched_ref(
                || -> TractOnnxRunner {
                    TractOnnxRunner::new(CLASSIFIER_PATH_ONNX, &ball_model_input_shape).unwrap()
                },
                |runner| runner.run_inference_single_io(ball_input_buffer).len(),
                BatchSize::SmallInput,
            )
        },
    );
    // TODO add the models
    // group.bench_with_input(
    //     BenchmarkId::new("TractOnnxRunner", "semseg"),
    //     &semseg_image,
    //     |b, image| {
    //         b.iter_batched_ref(
    //             || -> (TractOnnxRunner, Vec<f32>) {
    //                 (
    //                     TractOnnxRunner::new(
    //                         SEMANTIC_SEGMENTATION_DEEPLAB_ONNX,
    //                         &deeplab_input_shape,
    //                     )
    //                     .unwrap(),
    //                     image
    //                         .resize_exact(
    //                             deeplab_input_shape[1] as u32,
    //                             deeplab_input_shape[2] as u32,
    //                             imageops::Triangle,
    //                         )
    //                         .as_bytes()
    //                         .iter()
    //                         .map(|p| *p as f32)
    //                         .collect_vec(),
    //                 )
    //             },
    //             |(runner, input_buffer)| runner.run_inference_single_io(input_buffer).len(),
    //             BatchSize::SmallInput,
    //         )
    //     },
    // );

    cfg_if::cfg_if! {
        if #[cfg(feature="tflite")] {
            group.bench_with_input(
                BenchmarkId::new("TfLiteRunner HULKS", BALL_SAMPLE_PATH),
                &ball_input_buffer,
                |b, input_buffer| {
                    b.iter_batched_ref(
                        || -> TfLiteRunner {
                            TfLiteRunner::new(_CLASSIFIER_PATH_TFLITE, _thread_count).unwrap()
                        },
                        |runner| runner.run_inference_single_io(input_buffer).len(),
                        BatchSize::SmallInput,
                    )
                },
            );
            // TODO add the models
            // group.bench_with_input(
            //     BenchmarkId::new("TfLiteRunner Deeplab", _SEMSEG_SAMPLE_PATH),
            //     &semseg_image,
            //     |b, image| {
            //         b.iter_batched_ref(
            //             || -> (TfLiteRunner,Vec<f32>) {
            //                 (
            //                     TfLiteRunner::new(_SEMANTIC_SEGMENTATION_DEEPLABL_TFLITE, _thread_count).unwrap(),
            //                     image
            //                     .resize_exact(
            //                         deeplab_input_shape[1] as u32,
            //                         deeplab_input_shape[2] as u32,
            //                         imageops::Triangle,
            //                     )
            //                     .as_bytes()
            //                     .iter()
            //                     .map(|p| *p as f32)
            //                     .collect_vec(),
            //                 )
            //             },
            //             |(runner, input_buffer)| runner.run_inference_single_io(input_buffer).len(),
            //             BatchSize::SmallInput,
            //         )
            //     },
            // );
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_benchmark
}

criterion_main!(benches);
