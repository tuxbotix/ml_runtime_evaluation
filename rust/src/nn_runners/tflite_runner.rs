// tflitec library has issues with cross compiling via Bazel right now.
// Until this is fixed, only host will be used.
cfg_if::cfg_if! {
if #[cfg(feature="tflite")] {

use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;
use tflitec::tensor::DataType;

use super::runner_traits::Runner;

pub struct TfLiteRunner {
    // The model must live at least as long as the interpreter in C or C++, not sure the status here.
    // model: Model,
    interpreter: Interpreter,
    current_first_output: Vec<f32>,
}

impl TfLiteRunner {
    pub fn new(network_path_tflite: &str, thread_count: i32) -> anyhow::Result<Self> {
        // Create interpreter options
        let options = Options { thread_count,is_xnnpack_enabled:true };

        let model = Model::new(network_path_tflite)?;
        // Create interpreter
        let interpreter = Interpreter::new(&model, Some(options))?;

        // Allocate tensors if you just created Interpreter or resized its inputs
        interpreter.allocate_tensors()?;

        let current_first_output = vec![0.0f32; interpreter.output(0).unwrap().data::<f32>().len()];
        Ok(Self {
            // model,
            interpreter,
            current_first_output,
        })
    }
}

impl Runner for TfLiteRunner {
    fn run_inference_single_io(&mut self, input_buffer: &[f32]) -> &[f32] {
        type InputOutputDatatype = f32;

        let input_output_index = 0;

        // let interpreter_input_tensor = &self.interpreter.input(input_output_index).unwrap();
        // // TODO use generics or input buffers should be "variant" style data.
        // assert_eq!(
        //     interpreter_input_tensor.data_type(),
        //     tensor::DataType::Float32
        // );

        // let input_buffer_len = input_buffer.len();
        // let interpreter_input_buffer_len = interpreter_input_tensor.data::<InputOutputDatatype>().len();
        // assert!(
        //     input_buffer_len == interpreter_input_buffer_len,
        //     "Input buffer size must match what is expected by the interpreter. input_buffer_len: {input_buffer_len} interpreter_input_buffer_len: {interpreter_input_buffer_len}"
        // );

        // Copy input to buffer of first tensor (with index 0)
        // Set data using Tensor handle if you have it already
        // assert!(interpreter_input_tensor.set_data(input_buffer).is_ok(), "Failed to set interpreter input buffer with input data");
        // Or set data using Interpreter:
        assert!(self.interpreter.copy(input_buffer, input_output_index).is_ok(),"Failed to set interpreter input buffer with input data");

        // // Invoke interpreter
        assert!(self.interpreter.invoke().is_ok(), "Inference failed!");

        let tensor = &self.interpreter.output(input_output_index).unwrap();

        // Get output tensor
        assert!(tensor.data_type() == DataType::Float32, "Output datatype is not float. Other types are not supported yet.");
        let output_len: usize = tensor.data::<InputOutputDatatype>().len();

        // // This shouldn't happen as the output is allocated on startup
        if self.current_first_output.len() < output_len {
            self.current_first_output.resize(output_len, 0.0f32);
        }
        self.current_first_output.copy_from_slice(tensor.data());
        self.current_first_output.as_slice()
    }

    fn get_input_shape(&self, _index: usize) -> Vec<usize> {
        todo!();
    }

    fn get_output_shape(&self, _index: usize) -> Vec<usize> {
        todo!();
    }
}

}
}
