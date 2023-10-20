use rulinalg::matrix;
use rulinalg::matrix::Matrix;

mod helpers;
use helpers::sigmoid;

fn main() {
    let inputs: Matrix<f32> = matrix![18.4;
                        0.0];
    let hidden_layer_weights: Matrix<f32> = matrix![1.0, 0.5;
                                    0.5, 1.0];
    let hidden_layer_biases: Matrix<f32> = matrix![1.1;
                                    0.72];
    let output_weights: Matrix<f32> = matrix![1.1, 0.72];
    let output_biases: Matrix<f32> = matrix![1.1];

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

    dbg!(&output);
}
