mod compiled_nn_runner;
mod onnx_runtime_runner;
mod runner_traits;
mod tract_onnx_runner;

pub use compiled_nn_runner::CompiledNNRunner;
pub use runner_traits::Runner;
pub use tract_onnx_runner::TractOnnxRunner;

// tflitec library has issues with cross compiling via Bazel right now.
// Until this is fixed, only host will be used.
cfg_if::cfg_if! {
    if #[cfg(feature="tflite")] {
        mod tflite_runner;
        pub use tflite_runner::TfLiteRunner;
    }
}
