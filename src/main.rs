mod helpers;
use helpers::load_data;

mod network;
use network::Network;

fn main() {
    let training_data = load_data("mnist/train").expect("Data not loaded correctly.");
    let mut network = Network::new(vec![8, 8, 10], 784, 0.08);

    network.train(training_data);

    let accuracy_data = load_data("mnist/t10k").expect("Data not loaded correctly.");
    let mut correct: i32 = 0;
    let total = accuracy_data.len();
    for data in accuracy_data.into_iter() {
        let correct_val = &data.classification;
        let calc = network.feed_forward(data.inputs);
        let mut guess: u8 = 0;
        for i in 0..calc.len() {
            if calc[i] > calc[guess as usize] {
                guess = i as u8;
            }
        }
        if &guess == correct_val {
            correct += 1;
        }
    }
    println!("Accuracy: {} out of {}", correct, total);

    network
        .output_data("network-data/data")
        .expect("Outputing to file failed.");
}

// let weather_code_normalize = |x: f32| -> f32 { x / 100.0 };
//     let temperature_normalize = |x: f32| -> f32 { x / 35.0 };
//     let mut data = vec![
//         TrainingData {
//             inputs: vec![temperature_normalize(18.4), weather_code_normalize(0.0)],
//             target: vec![1.0, 0.0],
//         },
//         TrainingData {
//             inputs: vec![temperature_normalize(20.4), weather_code_normalize(5.0)],
//             target: vec![1.0, 0.0],
//         },
//         TrainingData {
//             inputs: vec![temperature_normalize(14.4), weather_code_normalize(15.0)],
//             target: vec![1.0, 0.0],
//         },
//         TrainingData {
//             inputs: vec![temperature_normalize(10.4), weather_code_normalize(43.0)],
//             target: vec![0.7, 0.3],
//         },
//         TrainingData {
//             inputs: vec![temperature_normalize(32.2), weather_code_normalize(99.0)],
//             target: vec![0.0, 1.0],
//         },
//         TrainingData {
//             inputs: vec![temperature_normalize(0.0), weather_code_normalize(0.0)],
//             target: vec![0.0, 1.0],
//         },
//     ];
//     for data in data.iter() {
//         dbg!("before", network.feed_forward(data.inputs.clone()));
//     }
//     let mut training_data: Vec<TrainingData> = Vec::new();
//     for _ in 0..10000 {
//         training_data.append(&mut data);
//     }
//     network.train(training_data);
//     for data in data.iter() {
//         dbg!("after", network.feed_forward(data.inputs.clone()));
//     }
