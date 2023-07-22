use tch::{CModule, Tensor};

use super::runner_traits::Runner;

pub struct TorchRunner {
    // The model must live at least as long as the interpreter in C or C++, not sure the status here.
    // model: Model,
    model: CModule,
    current_first_output: Vec<f32>,
}

impl TorchRunner {
    pub fn new(network_path_tflite: &str) -> anyhow::Result<Self> {
        // Create interpreter options
        let mut model = tch::CModule::load(network_path_tflite)?;
        model.set_eval();

        let input_buffer: Vec<f64> = vec![];
        let tensor = vec![Tensor::from_slice(input_buffer.as_slice())];
        let output = model.forward_ts(&tensor).unwrap();
        let mut output_vec = vec![0.0; output.numel()];
        output.copy_data(output_vec.as_mut_slice(), output.numel());

        Ok(Self {
            model,
            current_first_output: Default::default(),
        })
    }
}

impl Runner for TorchRunner {
    fn run_inference_single_io(&mut self, input_buffer: &[f32]) -> &[f32] {
        // Multi input - single output mode. For fancier IO structures, use forward_is where an "ivalue" is passed.
        let tensors = vec![Tensor::from_slice(input_buffer)];
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
