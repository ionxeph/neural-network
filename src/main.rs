mod helpers;
use helpers::load_data;

mod network;
use network::{Network, NetworkData};
use serde_json::Error;

const LEARNING_RATE: f64 = 0.3;

fn main() -> std::result::Result<(), Error> {
    let network_data: Option<NetworkData> = match std::fs::read_to_string("network-data/data.json")
    {
        Ok(data) => Some(serde_json::from_str(&data)?),
        Err(_) => None,
    };
    let mut network: Network;
    if let Some(data) = network_data {
        network = Network::from_data(data, LEARNING_RATE)
    } else {
        network = Network::new(vec![16, 16, 10], 784, LEARNING_RATE);
    }

    let accuracy_data = load_data("mnist/t10k").expect("Data not loaded correctly.");
    let mut before: i32 = 0;
    let total = accuracy_data.len();
    for data in accuracy_data.clone().into_iter() {
        let correct_val = &data.classification;
        let calc = network.feed_forward(data.inputs);
        let mut guess: u8 = 0;
        for i in 0..calc.len() {
            if calc[i] > calc[guess as usize] {
                guess = i as u8;
            }
        }
        if &guess == correct_val {
            before += 1;
        }
    }
    println!("Before accuracy: {} out of {}", before, total);

    let training_data = load_data("mnist/train").expect("Data not loaded correctly.");
    network.train(training_data, 50, 1);
    // network.train(training_data.into_iter().skip(59990).collect(), 10, 1);

    let mut after: i32 = 0;
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
            after += 1;
        }
    }
    println!("After accuracy: {} out of {}", after, total);

    if after > before {
        println!("Model improved, saving to file.");
        let network_data = network.output_data();

        let json = serde_json::to_string(&network_data)?;

        std::fs::write("network-data/data.json", json).expect("Unable to write file");
    } else {
        println!("You fucked up, not saving to file.");
    }

    Ok(())
}
