use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use image::imageops;
use itertools::Itertools;
use nn_backend_test::nn_runners::TorchRunner;
use pprof::criterion::{Output, PProfProfiler};

extern crate nn_backend_test;
use crate::nn_backend_test::nn_runners::{CompiledNNRunner, Runner, TractOnnxRunner};

#[cfg(feature = "tflite")]
use crate::nn_backend_test::nn_runners::TfLiteRunner;

// TODO find a configuration based approach to load all the possible cases and benchmark! these hardcoded paths are messy.

#[derive(Debug)]
enum Runtimes {
    OnnxTract,
    Torch,
    TfLite,
    Tf,
    CompiledNN,
}

#[derive(Debug)]
enum ModelTypes {
    HulksBallModel,
    HulksMultiClassifier,
    SemsegDeeplab,
    YoloV8,
}

type InputAndShapeFn = dyn Fn() -> (Vec<f32>, Vec<u32>);

fn get_bench_combos() -> Vec<(Runtimes, ModelTypes, Box<InputAndShapeFn>, &'static str)> {
    vec![
        (
            Runtimes::CompiledNN,
            ModelTypes::HulksBallModel,
            Box::new(ball_data),
            "../models/hulks_2022/classifier_updated.hdf5",
        ),
        (
            Runtimes::CompiledNN,
            ModelTypes::HulksMultiClassifier,
            Box::new(ball_data),
            "../models/hulks_2022/classifier_multiclass.hdf5",
        ),
        (
            Runtimes::OnnxTract,
            ModelTypes::HulksBallModel,
            Box::new(ball_data),
            "../models/hulks_2022/classifier.onnx",
        ),
        // (
        //     Runtimes::OnnxTract,
        //     ModelTypes::SemsegDeeplab,
        //     Box::new(semseg_data),
        //     "../models/semantic_segmentation/lite-model_deeplabv3_1_metadata_2_converted.onnx",
        // ),
        // (
        //     Runtimes::TfLite,
        //     ModelTypes::HulksBallModel,
        //     "../models/hulks_2022/classifier.tflite",
        // ),
        // (
        //     Runtimes::OnnxTract,
        //     ModelTypes::YoloV8,
        //     Box::new(yolo_v8_data),
        //     "../models/yolo/yolov8n.onnx",
        // ),
        (
            Runtimes::Torch,
            ModelTypes::YoloV8,
            Box::new(yolo_v8_data),
            "../models/yolo/yolov8n.torchscript",
        ),
    ]
}

fn ball_data() -> (Vec<f32>, Vec<u32>) {
    // open image, resize it and make a Tensor out of it
    let ball_image = image::open("../data/ball_sample.png").unwrap().to_rgb8();
    // extract red channel only
    let ball_input_buffer = ball_image.pixels().map(|f| f[0] as f32).collect_vec();
    // options used by different backends
    let ball_model_input_shape = [ball_image.width(), ball_image.height(), 1];

    (ball_input_buffer, ball_model_input_shape.to_vec())
}

fn semseg_data() -> (Vec<f32>, Vec<u32>) {
    // TODO check channel order
    let deeplab_input_shape = [1, 257, 257, 3];
    // open image, resize it and make a Tensor out of it
    let semseg_image =
        image::open("../data/01024_GermanOpen2019_HULKs_Sabretooth-2nd_52240197_upper-002.png")
            .unwrap();

    let buffer = semseg_image
        .resize_exact(
            deeplab_input_shape[1] as u32,
            deeplab_input_shape[2] as u32,
            imageops::Triangle,
        )
        .as_bytes()
        .iter()
        .map(|p| *p as f32)
        .collect_vec();

    (buffer, deeplab_input_shape.to_vec())
}

fn yolo_v8_data() -> (Vec<f32>, Vec<u32>) {
    // TODO check channel order
    let yolo_v8_shape = [1, 3, 640, 640];
    // open image, resize it and make a Tensor out of it
    let semseg_image =
        image::open("../data/01024_GermanOpen2019_HULKs_Sabretooth-2nd_52240197_upper-002.png")
            .unwrap();

    let buffer = semseg_image
        .resize_exact(
            yolo_v8_shape[2] as u32,
            yolo_v8_shape[3] as u32,
            imageops::Triangle,
        )
        .as_bytes()
        .iter()
        .map(|p| *p as f32)
        .collect_vec();

    (buffer, yolo_v8_shape.to_vec())
}

fn criterion_benchmark(c: &mut Criterion) {
    let _thread_count = 1;

    let mut group = c.benchmark_group("NN Runner");

    for (runtime, model_type, input_provider, model_path) in get_bench_combos() {
        let (input_buffer, input_shape) = input_provider();

        group.bench_with_input(
            BenchmarkId::new(
                format!("model type: {:?}", model_type),
                format!("Runtime: {:?}", runtime),
            ),
            &input_buffer,
            |b, input_buffer| {
                b.iter_batched_ref(
                    || -> Box<dyn Runner> { get_runtime(&runtime, model_path, &input_shape) },
                    |runner| runner.run_inference_single_io(input_buffer).len(),
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn get_runtime(runtime: &Runtimes, model_path: &str, input_shape: &Vec<u32>) -> Box<dyn Runner> {
    match runtime {
        Runtimes::CompiledNN => Box::new(CompiledNNRunner::new(model_path).unwrap()),
        Runtimes::OnnxTract => Box::new(
            TractOnnxRunner::new(
                model_path,
                input_shape
                    .iter()
                    .map(|v| *v as usize)
                    .collect_vec()
                    .as_slice(),
            )
            .unwrap(),
        ),
        Runtimes::Torch => Box::new(
            TorchRunner::new(
                &model_path,
                input_shape
                    .iter()
                    .map(|v| *v as usize)
                    .collect_vec()
                    .as_slice(),
            )
            .unwrap(),
        ),
        Runtimes::TfLite => todo!(),
        Runtimes::Tf => todo!(),
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_benchmark
}

criterion_main!(benches);
