use rand::Rng;
use rulinalg::matrix;
use rulinalg::matrix::{BaseMatrix, Matrix};

mod helpers;
use helpers::{d_sigmoid, get_weight_delta, sigmoid};

fn main() {
    let target: f32 = 1.0;
    let learning_rate: f32 = 2.0;
    let inputs: Matrix<f32> = matrix![sigmoid(18.4);
    sigmoid(0.0)];
    let mut rng = rand::thread_rng();

    let hidden_layer_weights: Matrix<f32> = matrix![rng.gen(), rng.gen();
    rng.gen(), rng.gen()];
    let hidden_layer_biases: Matrix<f32> = matrix![rng.gen();
    rng.gen()];
    let output_weights: Matrix<f32> = matrix![rng.gen(), rng.gen()];
    let output_biases: Matrix<f32> = matrix![rng.gen()];

    let hidden_layer_output: Matrix<f32> = Matrix::new(
        2,
        1,
        (&hidden_layer_weights * &inputs + &hidden_layer_biases)
            .into_vec()
            .iter()
            .map(|x| sigmoid(*x))
            .collect::<Vec<f32>>(),
    );

    let output: Matrix<f32> = Matrix::new(
        1,
        1,
        (&output_weights * &hidden_layer_output + &output_biases)
            .into_vec()
            .iter()
            .map(|x| sigmoid(*x))
            .collect::<Vec<f32>>(),
    );

    dbg!(&output.data()[0]);
    dbg!((target - output.data()[0]).powf(2.0) / 2.0); // cost

    let output_gradient: Matrix<f32> = Matrix::new(
        1,
        1,
        (output)
            .into_vec()
            .iter()
            .map(|output: &f32| (target - *output) * d_sigmoid(*output) * learning_rate)
            .collect::<Vec<f32>>(),
    );

    let hidden_layer_gradient: Matrix<f32> = Matrix::new(
        2,
        1,
        (&output_weights.transpose() * &output_gradient)
            .into_vec()
            .iter()
            .map(|x: &f32| d_sigmoid(*x) * learning_rate)
            .collect::<Vec<f32>>(),
    );

    let new_output_bias = &output_biases + &output_gradient;

    let new_output_weights =
        &output_weights + get_weight_delta(&hidden_layer_output, &output_gradient);

    let new_hidden_layer_biases = &hidden_layer_biases + &hidden_layer_gradient;

    let new_hidden_layer_weights =
        &hidden_layer_weights + get_weight_delta(&inputs, &hidden_layer_gradient);

    let new_hidden_layer_output: Matrix<f32> = Matrix::new(
        2,
        1,
        (&new_hidden_layer_weights * &inputs + &new_hidden_layer_biases)
            .into_vec()
            .iter()
            .map(|x| sigmoid(*x))
            .collect::<Vec<f32>>(),
    );

    let new_output: Matrix<f32> = Matrix::new(
        1,
        1,
        (&new_output_weights * &new_hidden_layer_output + &new_output_bias)
            .into_vec()
            .iter()
            .map(|x| sigmoid(*x))
            .collect::<Vec<f32>>(),
    );

    dbg!(&new_output.data()[0]);
    dbg!((target - new_output.data()[0]).powf(2.0) / 2.0); // cost
}
