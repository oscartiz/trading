use candle_core::{Device, Tensor, DType, Result};
use candle_nn::{Conv2dConfig, Conv2d, VarBuilder, Module};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::zeros((16, 1, 1, 2), DType::F32, &device)?;
    let bias = Tensor::zeros(16, DType::F32, &device)?;
    
    // Test if we can pass [1, 2] to conv2d
    let x = Tensor::zeros((1, 1, 100, 40), DType::F32, &device)?;
    
    // Symmetric stride works
    let _ = x.conv2d(&weight, 0, 1, 1, 1)?;
    println!("Symmetric stride works");
    
    // Can we do asymmetric?
    // Let's check Tensor::conv2d signature again.
    // In candle 0.8 it seems to be symmetric in the high level API.
    
    Ok(())
}
