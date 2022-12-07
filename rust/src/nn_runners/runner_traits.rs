pub trait Runner {
    fn run_inference_single_io(&mut self, input_buffer: &[f32]) -> &[f32];
}
