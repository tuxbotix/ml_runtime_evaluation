use std::time::Instant;

use tract_core::internal::*;
use tract_onnx::prelude::*;

use super::runner_traits::Runner;

type SimplePlanTyped = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
>;

#[derive(Clone)]
pub struct TractOnnxRunner {
    // model:TypedModel,
    model: SimplePlanTyped,
    output: Vec<f32>,
    // input_shape: Vec<usize>,
    input_tensor: Tensor,
}

impl TractOnnxRunner {
    pub fn new(network_path_onnx: &str, input_shape: &[usize]) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(network_path_onnx)?
            // specify input type and shape
            .with_input_fact(0, f32::fact(input_shape).into())?
            // this model hardcodes a "1" as batch output shape, erase the output shape
            // to let tract perform inference and find "N"
            .with_output_fact(0, InferenceFact::default())?
            // optimize the model
            .into_optimized()?
            // make the model runnable and fix its inputs and outputs
            .into_runnable()?;

        // Resize
        let output_len = model
            .model
            .output_fact(0)
            .unwrap()
            .shape
            .as_concrete()
            .unwrap()
            .iter()
            .copied()
            .reduce(std::ops::Mul::mul)
            .unwrap();

        let input_len = input_shape
            .iter()
            .copied()
            .reduce(std::ops::Mul::mul)
            .unwrap();

        let input_tensor = Tensor::from_shape(input_shape, vec![0.0f32; input_len].as_slice());

        let mut runner = Self {
            model,
            output: vec![0.0f32; output_len],
            input_tensor: input_tensor.unwrap(),
        };

        // Dummy run for first time.
        {
            // Create input
            let vec = vec![0.0f32; input_len];
            runner.run_inference_single_io(vec.as_slice());
        }
        Ok(runner)
    }
}

impl Runner for TractOnnxRunner {
    fn run_inference_single_io(&mut self, input_buffer: &[f32]) -> &[f32] {
        let input_output_index = 0;

        self.input_tensor
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(input_buffer);

        let result = self.model.run(tvec![self.input_tensor.clone()]).unwrap();

        let result_slice = &result[input_output_index].as_slice().unwrap();

        // this should ideally never occur
        if result_slice.len() >= self.output.len() {
            self.output.resize(result_slice.len(), 0.0f32);
        }

        self.output.copy_from_slice(result_slice);
        &self.output
    }
}
