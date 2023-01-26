pub trait Runner {
    fn run_inference_single_io(&mut self, input_buffer: &[f32]) -> &[f32];

    fn get_input_shape(&self, index: usize) -> Vec<usize>;
    fn get_output_shape(&self, index: usize) -> Vec<usize>;
}
