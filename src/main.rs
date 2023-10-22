mod helpers;

mod network;
use network::{Network, TrainingData};

fn main() {
    let mut network = Network::new(vec![3, 3, 1], 2, 0.2);
    let weather_code_normalize = |x: f32| -> f32 { x / 100.0 };
    let temperature_normalize = |x: f32| -> f32 { x / 35.0 };
    let mut data = vec![
        TrainingData {
            inputs: vec![temperature_normalize(18.4), weather_code_normalize(0.0)],
            target: 1.0,
        },
        TrainingData {
            inputs: vec![temperature_normalize(20.4), weather_code_normalize(5.0)],
            target: 1.0,
        },
        TrainingData {
            inputs: vec![temperature_normalize(14.4), weather_code_normalize(15.0)],
            target: 1.0,
        },
        TrainingData {
            inputs: vec![temperature_normalize(10.4), weather_code_normalize(43.0)],
            target: 0.7,
        },
        TrainingData {
            inputs: vec![temperature_normalize(32.2), weather_code_normalize(99.0)],
            target: 0.0,
        },
        TrainingData {
            inputs: vec![temperature_normalize(0.0), weather_code_normalize(0.0)],
            target: 0.0,
        },
    ];
    for data in data.iter() {
        dbg!("before", network.feed_forward(data.inputs.clone()));
    }
    let mut training_data: Vec<TrainingData> = Vec::new();
    for _ in 0..10000 {
        training_data.append(&mut data);
    }
    network.train(training_data);
    for data in data.iter() {
        dbg!("after", network.feed_forward(data.inputs.clone()));
    }
}
