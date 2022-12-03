use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;
use tflitec::tensor;

fn loat_tflite_model() -> anyhow::Result<Vec<f32>> {
    // Create interpreter options
    let mut options = Options::default();
    options.thread_count = 1;

    // Load example model which outputs y = 3 * x
    let model = Model::new("neural_networks/mnist_model.tflite")?;
    // Create interpreter
    let interpreter = Interpreter::new(&model, Some(options))?;
    // Resize input
    let input_shape = tensor::Shape::new(vec![10, 8, 8, 3]);
    let input_element_count = input_shape
        .dimensions()
        .iter()
        .copied()
        .reduce(std::ops::Mul::mul)
        .unwrap();
    interpreter.resize_input(0, input_shape)?;
    // Allocate tensors if you just created Interpreter or resized its inputs
    interpreter.allocate_tensors()?;

    // Create dummy input
    let data = (0..input_element_count)
        .map(|x| x as f32)
        .collect::<Vec<f32>>();

    let input_tensor = interpreter.input(0)?;
    assert_eq!(input_tensor.data_type(), tensor::DataType::Float32);

    // Copy input to buffer of first tensor (with index 0)
    // You have 2 options:
    // Set data using Tensor handle if you have it already
    assert!(input_tensor.set_data(&data[..]).is_ok());
    // Or set data using Interpreter:
    assert!(interpreter.copy(&data[..], 0).is_ok());

    // Invoke interpreter
    assert!(interpreter.invoke().is_ok());

    // Get output tensor
    let output_tensor = interpreter.output(0)?;

    assert_eq!(output_tensor.shape().dimensions(), &vec![10, 8, 8, 3]);
    let output_vector = output_tensor.data::<f32>().to_vec();
    let expected: Vec<f32> = data.iter().map(|e| e * 3.0).collect();
    assert_eq!(expected, output_vector);

    Ok(output_vector)
}
