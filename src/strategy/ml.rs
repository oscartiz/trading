use candle_core::{Result, Tensor, Module, ModuleT};
use candle_nn::{BatchNorm, Conv2d, Linear, LSTM, VarBuilder, RNN};

// We need to implement the DeepLOB architecture.
// CNN Blocks
// Inception Module
// LSTM
// FC

#[derive(Debug)]
pub struct InceptionModule {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    bn: BatchNorm,
}

fn conv2d_v2(in_channels: usize, out_channels: usize, kernel_size: (usize, usize), cfg: candle_nn::Conv2dConfig, vb: candle_nn::VarBuilder) -> Result<Conv2d> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let weight = vb.get_with_hints(
        (out_channels, in_channels, kernel_size.0, kernel_size.1),
        "weight",
        init_ws,
    )?;
    let bias = vb.get(out_channels, "bias").ok();
    Ok(Conv2d::new(weight, bias, cfg))
}

impl InceptionModule {
    pub fn load(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv1 = candle_nn::conv2d(in_channels, out_channels, 1, Default::default(), vb.pp("conv1"))?;
        
        // For conv2 and conv3, we use padding 0 here and manual pad_with_zeros() in forward()
        let conv2 = conv2d_v2(in_channels, out_channels, (1, 3), Default::default(), vb.pp("conv2"))?;
        let conv3 = conv2d_v2(in_channels, out_channels, (1, 5), Default::default(), vb.pp("conv3"))?;
        
        let bn = candle_nn::batch_norm(out_channels * 3, 1e-5, vb.pp("bn"))?;
        
        Ok(Self { conv1, conv2, conv3, bn })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.conv1.forward(x)?;
        
        // Manual padding for conv2 (kernel 3): pad W by 1 on each side
        let x2_in = x.pad_with_zeros(3, 1, 1)?;
        let x2 = self.conv2.forward(&x2_in)?;
        
        // Manual padding for conv3 (kernel 5): pad W by 2 on each side
        let x3_in = x.pad_with_zeros(3, 2, 2)?;
        let x3 = self.conv3.forward(&x3_in)?;
        
        // Concat along channel dimension (dim=1)
        let out = Tensor::cat(&[&x1, &x2, &x3], 1)?;
        let out = self.bn.forward_t(&out, false)?;
        
        // Leaky ReLU (0.01 slope)
        let out = candle_nn::ops::leaky_relu(&out, 0.01)?;
        Ok(out)
    }
}

pub struct DeepLOB {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    conv3: Conv2d,
    bn3: BatchNorm,
    inception1: InceptionModule,
    inception2: InceptionModule,
    lstm: LSTM,
    fc: Linear,
}

impl DeepLOB {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        // Architecture from PyTorch:
        // conv_channels=(1, 32, 32, 32), inception_channels=64, lstm_hidden=64
        
        // conv1: kernel=(1, 2), stride=(1, 2)
        let conv1 = conv2d_v2(1, 32, (1, 2), Default::default(), vb.pp("conv1"))?;
        let bn1 = candle_nn::batch_norm(32, 1e-5, vb.pp("bn1"))?;
        
        // conv2: kernel=(1, 2), stride=(1, 2)
        let conv2 = conv2d_v2(32, 32, (1, 2), Default::default(), vb.pp("conv2"))?;
        let bn2 = candle_nn::batch_norm(32, 1e-5, vb.pp("bn2"))?;
        
        // conv3: kernel=(1, 10), stride=(1, 1)
        let conv3 = conv2d_v2(32, 32, (1, 10), Default::default(), vb.pp("conv3"))?;
        let bn3 = candle_nn::batch_norm(32, 1e-5, vb.pp("bn3"))?;
        
        let inception1 = InceptionModule::load(vb.pp("inception1"), 32, 64)?;
        let inception2 = InceptionModule::load(vb.pp("inception2"), 64 * 3, 64)?;
        
        // LSTM config: input=192, hidden=64
        let lstm = candle_nn::lstm(64 * 3, 64, Default::default(), vb.pp("lstm"))?;
        let fc = candle_nn::linear(64, 3, vb.pp("fc"))?;
        
        Ok(Self { conv1, bn1, conv2, bn2, conv3, bn3, inception1, inception2, lstm, fc })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        
        // CNN block 1
        let x = self.conv1.forward(x)?;
        // Emulate stride (1, 2) along W dimension (dim 3)
        let indices1 = Tensor::from_vec((0..20).map(|i| (i * 2) as u32).collect::<Vec<_>>(), 20, device)?;
        let x = x.index_select(&indices1, 3)?;
        let x = self.bn1.forward_t(&x, false)?;
        let x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        
        // CNN block 2
        let x = self.conv2.forward(&x)?;
        // Emulate stride (1, 2) along W dimension
        let indices2 = Tensor::from_vec((0..10).map(|i| (i * 2) as u32).collect::<Vec<_>>(), 10, device)?;
        let x = x.index_select(&indices2, 3)?;
        let x = self.bn2.forward_t(&x, false)?;
        let x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        
        // CNN block 3
        let x = self.conv3.forward(&x)?;
        let x = self.bn3.forward_t(&x, false)?;
        let x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        
        // Inception blocks
        let x = self.inception1.forward(&x)?;
        let x = self.inception2.forward(&x)?;
        
        // Prepare for LSTM: (B, C, T, W) -> (B, T, C)
        // x should be (B, 192, 100, 1) after conv3 + inceptions
        let x = x.squeeze(3)?.transpose(1, 2)?;
        
        // LSTM forward
        let lstm_out = self.lstm.seq(&x)?; // shape: (B, T, hidden)
        
        // Take last hidden state
        let last_hidden = lstm_out.last().unwrap().h();
        
        // FC
        let logits = self.fc.forward(&last_hidden)?;
        Ok(logits)
    }

    pub fn predict(&self, x: &Tensor) -> Result<Vec<f32>> {
        let logits = self.forward(x)?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        // Convert [1, 3] tensor to Vec<f32>
        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;
        Ok(probs_vec)
    }
}
