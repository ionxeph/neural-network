pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (std::f64::consts::E as f32).powf(-x))
}
