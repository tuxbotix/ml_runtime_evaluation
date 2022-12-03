// use std::path::Path;

// use crate::Instant;
// use mcai_onnxruntime::{
//     environment::{EnvBuilder, Environment},
//     session::Session,
//     tensor::{FromArray, InputTensor, OrtOwnedTensor},
//     *,
// };

// use super::runner_traits::Runner;

// pub struct OnnxrRuntimeRunner<'a> {
//     environment: Environment,
//     session: Session<'a>,
// }

// impl OnnxrRuntimeRunner<'_> {
//     pub fn new(network_path_onnx: &str, input_shape: &[i32]) -> anyhow::Result<Self> {
//         let environment = Environment::builder()
//             .with_name("test")
//             .with_log_level(LoggingLevel::Verbose)
//             .build()?;

//         let mut session = environment
//             .new_session_builder()?
//             .with_optimization_level(GraphOptimizationLevel::Extended)?
//             .with_number_threads(1)?
//             .with_model_from_file(AsRef::<Path>::as_ref(network_path_onnx.clone()))?;

//         Ok(Self {
//             environment,
//             session,
//         })
//     }
// }

// impl Runner for OnnxrRuntimeRunner<'_> {
//     fn run_inference_single_io(&self, input_buffer: &[f32]) -> &[f32] {
//         let input_output_index = 0;

//         let input_shape = self.session.inputs[input_output_index].dimensions;
//                // Multiple inputs and outputs are possible
//         let input_tensor = vec![InputTensor::from_array(array)];

//         // input_tensor.
//         println!("OnnxRuntime sims: {:?}", input_shape.collect_vec());

//         let start: Instant = Instant::now();
//         let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(input_tensor).unwrap();
//         let onnx_runtime_duration = start.elapsed();

//         println!("OnnxRuntime results: {:?}", outputs);
//         println!("OnnxRuntime time: {:?}", onnx_runtime_duration);

//         outputs[0].as_slice()
//     }
// }
