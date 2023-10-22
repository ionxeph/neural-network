use rand::Rng;
use rulinalg::matrix::{BaseMatrix, Matrix};
use serde::{Deserialize, Serialize};
use serde_json::Error;

use crate::helpers;
use helpers::{d_sigmoid, get_weight_delta, sigmoid};

#[derive(Debug)]
pub struct Network {
    weights: Vec<Matrix<f32>>,
    biases: Vec<Matrix<f32>>,
    learning_rate: f32,
}

impl Network {
    pub fn new(layers: Vec<usize>, number_of_inputs: usize, learning_rate: f32) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights: Vec<Matrix<f32>> = Vec::with_capacity(layers.len());
        let mut biases: Vec<Matrix<f32>> = Vec::with_capacity(layers.len());
        let mut prev_layer_outputs = number_of_inputs;
        for layer in layers {
            weights.push(Matrix::from_fn(layer, prev_layer_outputs, |_, _| {
                rng.gen::<f32>() - rng.gen::<f32>()
            }));
            biases.push(Matrix::from_fn(layer, 1, |_, _| {
                rng.gen::<f32>() - rng.gen::<f32>()
            }));
            prev_layer_outputs = layer;
        }
        Network {
            weights,
            biases,
            learning_rate,
        }
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        if self.weights.is_empty() {
            panic!("No starting weights set!");
        }
        if self.biases.is_empty() {
            panic!("No starting biases set!");
        }
        if inputs.len() != self.weights[0].cols() {
            panic!("Inputs length needs to be {}", self.weights[0].cols());
        }

        let mut layer_output: Matrix<f32> = Matrix::new(inputs.len(), 1, inputs);

        for layer in 0..self.biases.len() {
            layer_output = Matrix::new(
                self.biases[layer].rows(),
                1,
                (&self.weights[layer] * layer_output + &self.biases[layer])
                    .into_vec()
                    .iter()
                    .map(|x| sigmoid(*x))
                    .collect::<Vec<f32>>(),
            );
        }
        layer_output.into_vec()
    }

    pub fn train(&mut self, training_data: Vec<TrainingData>) {
        for data in training_data.into_iter() {
            // feed-forward
            let mut layer_output: Vec<Matrix<f32>> =
                vec![Matrix::new(data.inputs.len(), 1, data.inputs)];
            for layer in 0..self.biases.len() {
                layer_output.push(Matrix::new(
                    self.biases[layer].rows(),
                    1,
                    (&self.weights[layer] * &layer_output[layer] + &self.biases[layer])
                        .into_vec()
                        .iter()
                        .map(|x| sigmoid(*x))
                        .collect::<Vec<f32>>(),
                ));
            }

            // back-propagate
            let mut layer_gradients: Vec<Matrix<f32>> =
                vec![Matrix::<f32>::zeros(1, 1); layer_output.len()];
            // last layer, aka output layer, gets special calculation
            let last_layer_index = layer_output.len() - 1;
            layer_gradients[last_layer_index] = Matrix::new(
                layer_output[last_layer_index].rows(),
                1,
                (layer_output[last_layer_index].clone())
                    .into_vec()
                    .iter()
                    .enumerate()
                    .map(|(i, output)| {
                        (data.target[i] - *output) * d_sigmoid(*output) * self.learning_rate
                    })
                    .collect::<Vec<f32>>(),
            );
            // will need to skip layer_gradients[0] as that's the inputs
            for i in (1..last_layer_index).rev() {
                layer_gradients[i] = Matrix::new(
                    layer_output[i].rows(),
                    1,
                    (self.weights[i].transpose() * &layer_gradients[i + 1])
                        .into_vec()
                        .iter()
                        .map(|x: &f32| d_sigmoid(*x) * self.learning_rate)
                        .collect::<Vec<f32>>(),
                );
            }

            for i in 0..self.biases.len() {
                self.biases[i] = &self.biases[i] + &layer_gradients[i + 1];
            }
            for i in 0..self.weights.len() {
                self.weights[i] =
                    &self.weights[i] + get_weight_delta(&layer_output[i], &layer_gradients[i + 1]);
            }
        }
    }

    pub fn output_data(&self, file_name: &str) -> std::result::Result<(), Error> {
        let mut weights: Vec<WeightData> = Vec::new();
        let mut biases: Vec<BiasData> = Vec::new();
        for layer in 0..self.biases.len() {
            let weight_rows = self.weights[layer].rows();
            let weight_cols = self.weights[layer].cols();
            let weight_data = self.weights[layer].clone().into_vec();
            let bias_rows = self.biases[layer].rows();
            let bias_data = self.biases[layer].clone().into_vec();
            weights.push(WeightData {
                rows: weight_rows,
                cols: weight_cols,
                data: weight_data,
            });
            biases.push(BiasData {
                rows: bias_rows,
                data: bias_data,
            });
        }
        let data = NetworkData { weights, biases };

        let json = serde_json::to_string(&data)?;

        std::fs::write(format!("{}.json", file_name), json).expect("Unable to write file");

        Ok(())
    }

    pub fn from_data(data: NetworkData, learning_rate: f32) -> Self {
        let mut weights: Vec<Matrix<f32>> = Vec::with_capacity(data.weights.len());
        let mut biases: Vec<Matrix<f32>> = Vec::with_capacity(data.biases.len());
        for layer in 0..data.weights.len() {
            weights.push(Matrix::new(
                data.weights[layer].rows,
                data.weights[layer].cols,
                data.weights[layer].data.clone(),
            ));
            biases.push(Matrix::new(
                data.biases[layer].rows,
                1,
                data.biases[layer].data.clone(),
            ));
        }
        Network {
            weights,
            biases,
            learning_rate,
        }
    }
}

#[derive(Clone)]
pub struct TrainingData {
    pub inputs: Vec<f32>,
    pub target: Vec<f32>,
    pub classification: u8,
}

#[derive(Serialize, Deserialize)]
pub struct NetworkData {
    weights: Vec<WeightData>,
    biases: Vec<BiasData>,
}

#[derive(Serialize, Deserialize)]
struct WeightData {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct BiasData {
    rows: usize,
    // cols is always 1
    data: Vec<f32>,
}
