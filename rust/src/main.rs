use itertools::Itertools;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

extern crate nn_backend_test;
use crate::nn_backend_test::nn_runners::{CompiledNNRunner, Runner, TractOnnxRunner};

#[cfg(tflite)]
use crate::nn_backend_test::nn_runners::TfLiteRunner;

// NN paths.

const CLASSIFIER_PATH: &str = "../models/hulks_2022/classifier.hdf5";
const CLASSIFIER_PATH_ONNX: &str = "../models/hulks_2022/classifier.onnx";
const _CLASSIFIER_PATH_TFLITE: &str = "../models/hulks_2022/classifier.tflite";

const BALL_SAMPLE_PATH: &str = "../data/ball_sample.png";

type RunnerAndResults<'a> = (Box<dyn Runner>, Vec<(&'a [f32], Duration)>);

fn main() {
    // open image, resize it and make a Tensor out of it
    let image = image::open(BALL_SAMPLE_PATH).unwrap().to_rgb8();

    // extract red channel only
    let input_buffer = image.pixels().map(|f| f[0] as f32).collect_vec();

    // options used by different backends
    let input_shape = [32, 32, 1];
    let _thread_count = 1;

    //
    let mut runner_result_map: HashMap<&str, RunnerAndResults> = HashMap::new();

    // Create runners.
    runner_result_map.insert(
        "compiled_nn",
        (
            Box::new(CompiledNNRunner::new(CLASSIFIER_PATH).unwrap()),
            Default::default(),
        ),
    );
    runner_result_map.insert(
        "tract",
        (
            Box::new(TractOnnxRunner::new(CLASSIFIER_PATH_ONNX, &input_shape).unwrap()),
            Default::default(),
        ),
    );
    // runner_result_map.insert(
    //     "onnx_runtime",
    //     (
    //         Box::new(OnnxrRuntimeRunner::new(CLASSIFIER_PATH_ONNX, &input_shape)),
    //         Default::default(),
    //     ),
    // );

    cfg_if::cfg_if! {
        if #[cfg(tflite)] {
            runner_result_map.insert(
                "tflite",
                (
                    Box::new(TfLiteRunner::new(_CLASSIFIER_PATH_TFLITE, _thread_count).unwrap()),
                    Default::default(),
                ),
            );
    }}

    println!("Runners are setup");

    // Run once per runtime.
    for (runner_name, (runner, result_and_duration_list)) in runner_result_map.iter_mut() {
        let start: Instant = Instant::now();

        let result = runner.as_mut().run_inference_single_io(&input_buffer);
        result_and_duration_list.push((result, start.elapsed()));

        let average_duration = get_average_duration(result_and_duration_list).as_micros();
        println!("Average runtime of:{runner_name} = {average_duration}us");
    }

    // println!("tractResultList: {:?}", tract_result_list);
}

fn get_average_duration(result_and_duration_list: &[(&[f32], Duration)]) -> Duration {
    result_and_duration_list
        .iter()
        .map(|(_, duration)| duration)
        .sum::<Duration>()
}
