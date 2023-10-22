use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use rulinalg::matrix::Matrix;
use std::{
    fs::File,
    io::{Cursor, Read},
};

use crate::network::TrainingData;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + std::f64::consts::E.powf(-x))
}

// derivative of sigmoid(x), where y is sigmoid(x)
pub fn d_sigmoid(y: f64) -> f64 {
    y * (1.0 - y)
}

pub fn get_weight_delta(m1: &Matrix<f64>, m2: &Matrix<f64>) -> Matrix<f64> {
    let m1 = m1.clone().into_vec();
    let m2 = m2.clone().into_vec();
    let mut result_arr: Vec<f64> = Vec::with_capacity(m1.len() * m2.len());
    (0..m2.len()).for_each(|i| {
        (0..m1.len()).for_each(|j| {
            //
            result_arr.push(m2[i] * m1[j]);
        });
    });

    Matrix::new(m2.len(), m1.len(), result_arr)
}

#[derive(Debug)]
pub struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

pub fn load_data(dataset_name: &str) -> Result<Vec<TrainingData>, std::io::Error> {
    let filename = format!("{}-labels-idx1-ubyte.gz", dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Vec<f64>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        images.push(image_data);
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let mut ret: Vec<TrainingData> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        let mut target = vec![0.0; 10];
        target[classification as usize] = 1.0;
        ret.push(TrainingData {
            inputs: image,
            target,
            classification,
        })
    }

    Ok(ret)
}
