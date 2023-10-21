use rulinalg::matrix::Matrix;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (std::f64::consts::E as f32).powf(-x))
}

// derivative of sigmoid(x), where y is sigmoid(x)
pub fn d_sigmoid(y: f32) -> f32 {
    y * (1.0 - y)
}

pub fn get_weight_delta(m1: &Matrix<f32>, m2: &Matrix<f32>) -> Matrix<f32> {
    let m1 = m1.clone().into_vec();
    let m2 = m2.clone().into_vec();
    let mut result_arr: Vec<f32> = Vec::with_capacity(m1.len() * m2.len());
    (0..m2.len()).for_each(|i| {
        (0..m1.len()).for_each(|j| {
            //
            result_arr.push(m2[i] * m1[j]);
        });
    });

    Matrix::new(m2.len(), m1.len(), result_arr)
}
