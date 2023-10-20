use rand::Rng;
use rulinalg::matrix;
use rulinalg::matrix::Matrix;

mod helpers;
use helpers::sigmoid;

fn main() {
    let target = 1.0;
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
        (hidden_layer_weights * inputs + hidden_layer_biases)
            .into_vec()
            .iter()
            .map(|x| sigmoid(*x))
            .collect::<Vec<f32>>(),
    );
    dbg!(&hidden_layer_output);

    let output: Matrix<f32> = Matrix::new(
        1,
        1,
        (output_weights * hidden_layer_output + output_biases)
            .into_vec()
            .iter()
            .map(|x| sigmoid(*x))
            .collect::<Vec<f32>>(),
    );

    dbg!(&output.data()[0]);
    dbg!((target - output.data()[0]).powf(2.0) / 2.0); // cost
}
