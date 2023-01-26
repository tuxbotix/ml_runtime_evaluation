use compiled_nn;

use super::runner_traits::Runner;

pub struct CompiledNNRunner {
    network_executor: compiled_nn::CompiledNN,
}

impl CompiledNNRunner {
    pub fn new(network_path_hdf: &str) -> anyhow::Result<Self> {
        let mut network_executor: compiled_nn::CompiledNN = Default::default();

        network_executor.compile(network_path_hdf);

        Ok(Self { network_executor })
    }
}

impl Runner for CompiledNNRunner {
    fn run_inference_single_io(&mut self, input_buffer: &[f32]) -> &[f32] {
        let input_output_index = 0;

        let nn_input: &mut [f32] = self.network_executor.input(input_output_index);

        assert!(
            input_buffer.len() <= nn_input.len(),
            "Input buffer is larger than the input available for compiledNN!"
        );

        nn_input.copy_from_slice(&input_buffer[..nn_input.len()]);

        // perform inference
        self.network_executor.apply();

        self.network_executor.output(input_output_index)
    }

    fn get_input_shape(&self, _index: usize) -> Vec<usize> {
        todo!();
    }

    fn get_output_shape(&self, _index: usize) -> Vec<usize> {
        todo!();
    }
}
