use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;
use tflitec::tensor::{self, DataType};

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
        let mut options = Options::default();
        options.thread_count = thread_count;

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

        let interpreter_input_tensor = &self.interpreter.input(input_output_index).unwrap();
        // TODO use generics or input buffers should be "variant" style data.
        assert_eq!(
            interpreter_input_tensor.data_type(),
            tensor::DataType::Float32
        );

        assert!(
            input_buffer.len() <= interpreter_input_tensor.data::<InputOutputDatatype>().len(),
            "Input buffer is larger than the input available for compiledNN!"
        );

        // Copy input to buffer of first tensor (with index 0)
        // Set data using Tensor handle if you have it already
        assert!(interpreter_input_tensor.set_data(&input_buffer).is_ok());
        // Or set data using Interpreter:
        // assert!(interpreter.copy(&data[..], 0).is_ok());

        // Invoke interpreter
        assert!(self.interpreter.invoke().is_ok());

        let tensor = &self.interpreter.output(input_output_index).unwrap();

        // Get output tensor
        assert!(tensor.data_type() == DataType::Float32);
        let output_len: usize = tensor.data::<InputOutputDatatype>().len();

        // This shouldn't happen as the output is allocated on startup
        if self.current_first_output.len() < output_len {
            self.current_first_output.resize(output_len, 0.0f32);
        }
        self.current_first_output.copy_from_slice(tensor.data());
        self.current_first_output.as_slice()
    }
}
