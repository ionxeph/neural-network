use std::{
    sync::{Arc, Mutex},
    thread,
};

use rand::Rng;
use rulinalg::matrix::{BaseMatrix, Matrix};
use serde::{Deserialize, Serialize};

use crate::helpers;
use helpers::{d_sigmoid, get_weight_delta, sigmoid};

// TODO: implement pruning
#[derive(Debug)]
pub struct Network {
    weights: Vec<Matrix<f64>>,
    biases: Vec<Matrix<f64>>,
    learning_rate: f64,
}

impl Network {
    pub fn new(layers: Vec<usize>, number_of_inputs: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights: Vec<Matrix<f64>> = Vec::with_capacity(layers.len());
        let mut biases: Vec<Matrix<f64>> = Vec::with_capacity(layers.len());
        let mut prev_layer_outputs = number_of_inputs;
        for layer in layers {
            weights.push(Matrix::from_fn(layer, prev_layer_outputs, |_, _| {
                rng.gen::<f64>() - rng.gen::<f64>()
            }));
            biases.push(Matrix::from_fn(layer, 1, |_, _| {
                rng.gen::<f64>() - rng.gen::<f64>()
            }));
            prev_layer_outputs = layer;
        }
        Network {
            weights,
            biases,
            learning_rate,
        }
    }

    pub fn feed_forward(&self, inputs: Vec<f64>) -> Vec<f64> {
        if self.weights.is_empty() {
            panic!("No starting weights set!");
        }
        if self.biases.is_empty() {
            panic!("No starting biases set!");
        }
        if inputs.len() != self.weights[0].cols() {
            panic!("Inputs length needs to be {}", self.weights[0].cols());
        }

        let mut layer_output: Matrix<f64> = Matrix::new(inputs.len(), 1, inputs);

        for layer in 0..self.biases.len() {
            layer_output = Matrix::new(
                self.biases[layer].rows(),
                1,
                (&self.weights[layer] * layer_output + &self.biases[layer])
                    .into_vec()
                    .iter()
                    .map(|x| sigmoid(*x))
                    .collect::<Vec<f64>>(),
            );
        }
        layer_output.into_vec()
    }

    pub fn train(&mut self, training_data: Vec<TrainingData>, batch_size: usize, epoch: usize) {
        let now = std::time::Instant::now();
        for epoch_i in 0..epoch {
            let mut summed_gradients: Arc<Mutex<Vec<Matrix<f64>>>> =
                Arc::new(Mutex::new(Vec::new()));
            let mut summed_weight_deltas: Arc<Mutex<Vec<Matrix<f64>>>> =
                Arc::new(Mutex::new(Vec::new()));
            let mut handles = vec![];
            for data in training_data.iter() {
                // feed-forward
                let mut layer_outputs: Vec<Matrix<f64>> =
                    vec![Matrix::new(data.inputs.len(), 1, data.inputs.clone())];
                for layer in 0..self.biases.len() {
                    layer_outputs.push(Matrix::new(
                        self.biases[layer].rows(),
                        1,
                        (&self.weights[layer] * &layer_outputs[layer] + &self.biases[layer])
                            .into_vec()
                            .iter()
                            .map(|x| sigmoid(*x))
                            .collect::<Vec<f64>>(),
                    ));
                }
                let layer_gradients: Vec<Matrix<f64>> =
                    self.calcualte_gradient(&layer_outputs, &data.target);

                let cloned_gradients = Arc::clone(&summed_gradients);
                let cloned_weights = Arc::clone(&summed_weight_deltas);
                let layer_count = self.weights.len();
                let handle = thread::spawn(move || {
                    let mut summed = cloned_gradients.lock().expect("Mutex mess.");
                    let mut weights = cloned_weights.lock().expect("Mutex mess.");

                    if summed.is_empty() {
                        for i in 0..layer_count {
                            (*weights)
                                .push(get_weight_delta(&layer_outputs[i], &layer_gradients[i + 1]));
                        }
                        *summed = layer_gradients;
                    } else {
                        for (i, gradient) in layer_gradients.into_iter().enumerate() {
                            if i > 0 {
                                (*weights)[i - 1] +=
                                    get_weight_delta(&layer_outputs[i - 1], &gradient);
                            }
                            (*summed)[i] += gradient;
                        }
                    }
                });
                handles.push(handle);

                if handles.len() == batch_size {
                    for handle in handles {
                        handle.join().expect("Threading mess.");
                    }
                    handles = Vec::new();
                    for i in 0..self.biases.len() {
                        self.biases[i] = &self.biases[i]
                            + &(*summed_gradients.lock().expect("Mutex mess."))[i + 1];
                    }
                    (0..self.weights.len()).for_each(|i| {
                        self.weights[i] = &self.weights[i]
                            + &(*summed_weight_deltas.lock().expect("Mutex mess."))[i];
                    });

                    summed_gradients = Arc::new(Mutex::new(Vec::new()));
                    summed_weight_deltas = Arc::new(Mutex::new(Vec::new()));
                }
            }
            println!("Completed epoch {} in {:.2?}", epoch_i + 1, now.elapsed());
        }
    }

    pub fn output_data(&self) -> NetworkData {
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
        NetworkData { weights, biases }
    }

    pub fn from_data(data: NetworkData, learning_rate: f64) -> Self {
        let mut weights: Vec<Matrix<f64>> = Vec::with_capacity(data.weights.len());
        let mut biases: Vec<Matrix<f64>> = Vec::with_capacity(data.biases.len());
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

    fn calcualte_gradient(
        &self,
        layer_outputs: &Vec<Matrix<f64>>,
        target: &[f64],
    ) -> Vec<Matrix<f64>> {
        let mut layer_gradients: Vec<Matrix<f64>> =
            vec![Matrix::<f64>::zeros(1, 1); layer_outputs.len()];
        // last layer, aka output layer, gets special calculation
        let last_layer_index = layer_outputs.len() - 1;
        layer_gradients[last_layer_index] = Matrix::new(
            layer_outputs[last_layer_index].rows(),
            1,
            (layer_outputs[last_layer_index].clone())
                .into_vec()
                .iter()
                .enumerate()
                .map(|(i, output)| (target[i] - *output) * d_sigmoid(*output) * self.learning_rate)
                .collect::<Vec<f64>>(),
        );
        // will need to skip layer_gradients[0] as that's the inputs
        for i in (1..last_layer_index).rev() {
            layer_gradients[i] = Matrix::new(
                layer_outputs[i].rows(),
                1,
                (self.weights[i].transpose() * &layer_gradients[i + 1])
                    .into_vec()
                    .iter()
                    .map(|x: &f64| d_sigmoid(*x) * self.learning_rate)
                    .collect::<Vec<f64>>(),
            );
        }
        layer_gradients
    }
}

// RELU training, doesn't quite work, need softmax for output layer probably
// impl Network {
//     pub fn feed_forward_relu(&self, inputs: Vec<f64>) -> Vec<f64> {
//         if self.weights.is_empty() {
//             panic!("No starting weights set!");
//         }
//         if self.biases.is_empty() {
//             panic!("No starting biases set!");
//         }
//         if inputs.len() != self.weights[0].cols() {
//             panic!("Inputs length needs to be {}", self.weights[0].cols());
//         }

//         let mut layer_output: Matrix<f64> = Matrix::new(inputs.len(), 1, inputs);

//         for layer in 0..self.biases.len() {
//             layer_output = Matrix::new(
//                 self.biases[layer].rows(),
//                 1,
//                 (&self.weights[layer] * layer_output + &self.biases[layer])
//                     .into_vec()
//                     .iter()
//                     .map(|x| relu(*x))
//                     .collect::<Vec<f64>>(),
//             );
//         }
//         layer_output.into_vec()
//     }

//     pub fn train_relu(
//         &mut self,
//         training_data: Vec<TrainingData>,
//         batch_size: usize,
//         epoch: usize,
//     ) {
//         let now = std::time::Instant::now();
//         let data_len = training_data.len();
//         let mut summed_layer_output: Vec<Matrix<f64>> = Vec::new();
//         let mut summed_target: Vec<f64> = Vec::new();
//         for epoch_i in 0..epoch {
//             for (idx, data) in training_data.iter().enumerate() {
//                 // feed-forward
//                 let mut layer_output: Vec<Matrix<f64>> =
//                     vec![Matrix::new(data.inputs.len(), 1, data.inputs.clone())];
//                 for layer in 0..self.biases.len() {
//                     layer_output.push(Matrix::new(
//                         self.biases[layer].rows(),
//                         1,
//                         (&self.weights[layer] * &layer_output[layer] + &self.biases[layer])
//                             .into_vec()
//                             .iter()
//                             .map(|x| relu(*x))
//                             .collect::<Vec<f64>>(),
//                     ));
//                 }
//                 if summed_layer_output.is_empty() {
//                     summed_layer_output = layer_output;
//                 } else {
//                     for (i, output) in layer_output.into_iter().enumerate() {
//                         summed_layer_output[i] += output;
//                     }
//                 }
//                 if summed_target.is_empty() {
//                     summed_target = data.target.clone();
//                 } else {
//                     for (i, target) in data.target.clone().into_iter().enumerate() {
//                         summed_target[i] += target;
//                     }
//                 }

//                 if idx > 0 && (idx + 1) % batch_size == 0 || idx == data_len {
//                     let mut avg_layer_output: Vec<Matrix<f64>> =
//                         Vec::with_capacity(summed_layer_output.len());
//                     for layer in summed_layer_output.into_iter() {
//                         avg_layer_output.push(Matrix::new(
//                             layer.rows(),
//                             1,
//                             layer
//                                 .into_vec()
//                                 .iter()
//                                 .map(|x| x / batch_size as f64)
//                                 .collect::<Vec<f64>>(),
//                         ));
//                     }
//                     let avg_target: Vec<f64> = summed_target
//                         .iter()
//                         .map(|target| target / batch_size as f64)
//                         .collect();

//                     // back-propagate
//                     let mut layer_gradients: Vec<Matrix<f64>> =
//                         vec![Matrix::<f64>::zeros(1, 1); avg_layer_output.len()];
//                     // last layer, aka output layer, gets special calculation
//                     let last_layer_index = avg_layer_output.len() - 1;
//                     layer_gradients[last_layer_index] = Matrix::new(
//                         avg_layer_output[last_layer_index].rows(),
//                         1,
//                         (avg_layer_output[last_layer_index].clone())
//                             .into_vec()
//                             .iter()
//                             .enumerate()
//                             .map(|(i, output)| {
//                                 (avg_target[i] - *output) * d_relu(*output) * self.learning_rate
//                             })
//                             .collect::<Vec<f64>>(),
//                     );
//                     // will need to skip layer_gradients[0] as that's the inputs
//                     for i in (1..last_layer_index).rev() {
//                         layer_gradients[i] = Matrix::new(
//                             avg_layer_output[i].rows(),
//                             1,
//                             (self.weights[i].transpose() * &layer_gradients[i + 1])
//                                 .into_vec()
//                                 .iter()
//                                 .map(|x: &f64| d_relu(*x) * self.learning_rate)
//                                 .collect::<Vec<f64>>(),
//                         );
//                     }

//                     for i in 0..self.biases.len() {
//                         self.biases[i] = &self.biases[i] + &layer_gradients[i + 1];
//                     }
//                     for i in 0..self.weights.len() {
//                         self.weights[i] = &self.weights[i]
//                             + get_weight_delta(&avg_layer_output[i], &layer_gradients[i + 1]);
//                     }

//                     summed_layer_output = Vec::new();
//                     summed_target = Vec::new();
//                 }
//             }
//             println!("Completed epoch {} in {:.2?}", epoch_i + 1, now.elapsed());
//         }
//     }
// }

#[derive(Clone)]
pub struct TrainingData {
    pub inputs: Vec<f64>,
    pub target: Vec<f64>,
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
    data: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct BiasData {
    rows: usize,
    // cols is always 1
    data: Vec<f64>,
}
