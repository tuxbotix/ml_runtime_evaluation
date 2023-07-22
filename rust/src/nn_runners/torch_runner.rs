use itertools::Itertools;
use tch::{CModule, Kind, Tensor};

use super::runner_traits::Runner;

pub struct TorchRunner {
    // The model must live at least as long as the interpreter in C or C++, not sure the status here.
    // model: Model,
    model: CModule,
    input_tensors: Vec<Tensor>,
    current_first_output: Vec<f32>,
}

impl TorchRunner {
    pub fn new(network_path_tflite: &str, input_shape: &[usize]) -> anyhow::Result<Self> {
        let device = tch::Device::Cpu;

        tch::jit::set_graph_executor_optimize(true);
        tch::jit::set_tensor_expr_fuser_enabled(true);

        // Create interpreter options
        let mut model = tch::CModule::load(network_path_tflite)?;
        model.set_eval();

        let shape = input_shape.iter().map(|v| *v as i64).collect_vec();

        let input_tensors = vec![Tensor::zeros(shape, (Kind::Float, device))];

        let output = model.forward_ts(&input_tensors).unwrap();
        let mut output_vec = vec![0.0f32; output.numel()];
        output.copy_data(output_vec.as_mut_slice(), output.numel());

        Ok(Self {
            model,
            input_tensors,
            current_first_output: Default::default(),
        })
    }
}

impl Runner for TorchRunner {
    fn run_inference_single_io(&mut self, input_buffer: &[f32]) -> &[f32] {
        // Multi input - single output mode. For fancier IO structures, use forward_is where an "ivalue" is passed.

        assert!(
            self.input_tensors[0].numel() == input_buffer.len(),
            "input buffer len {} and input tensor numl {} mismatch",
            input_buffer.len(),
            self.input_tensors[0].numel(),
        );

        let tensors = vec![Tensor::from_slice(input_buffer).reshape(self.input_tensors[0].size())];

        // let tensors = vec![Tensor::from_slice(input_buffer).reshape(self.input_tensors[0].size())];
        let output = self.model.forward_ts(&tensors).unwrap();

        self.current_first_output.resize(output.numel(), 0.0);
        output.copy_data(self.current_first_output.as_mut_slice(), output.numel());
        &self.current_first_output.as_slice()
    }

    fn get_input_shape(&self, _index: usize) -> Vec<usize> {
        todo!();
    }

    fn get_output_shape(&self, _index: usize) -> Vec<usize> {
        todo!();
    }
}
