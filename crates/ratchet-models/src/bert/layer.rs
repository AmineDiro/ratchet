use ratchet::{shape, Device, Tensor};
use ratchet_nn::{LayerNorm, Linear, Module};

use super::{attn::BertSelfAttention, mlp::MLP, model::BertConfig};

#[derive(Debug)]
pub struct BertEncoderLayer {
    attn: BertSelfAttention,
    mlp: MLP,
    norm: ratchet_nn::LayerNorm,
}
impl Module for BertEncoderLayer {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<ratchet::Tensor> {
        let mut x = self.attn.schedule(input)?;
        let residual = x.clone();
        x = self.mlp.schedule(x)?;
        x = self.norm.schedule(x.add(residual)?)?;
        Ok(x)
    }
}

impl BertEncoderLayer {
    pub fn new(config: &BertConfig, device: Device) -> Self {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let scale_val = 1.0 / (config.hidden_size as f32).sqrt();
        let softmax_scale = Tensor::from_data([scale_val], shape![1], device.clone());

        let attn = BertSelfAttention {
            q: Linear::new(
                Tensor::randn::<f32>(shape!(config.hidden_size, all_head_size), device.clone()),
                Some(Tensor::randn::<f32>(shape!(all_head_size), device.clone())),
            ),
            k: Linear::new(
                Tensor::randn::<f32>(shape!(config.hidden_size, all_head_size), device.clone()),
                Some(Tensor::randn::<f32>(shape!(all_head_size), device.clone())),
            ),
            v: Linear::new(
                Tensor::randn::<f32>(shape!(config.hidden_size, all_head_size), device.clone()),
                Some(Tensor::randn::<f32>(shape!(all_head_size), device.clone())),
            ),
            out: Linear::new(
                Tensor::randn::<f32>(
                    shape!(config.hidden_size, config.hidden_size),
                    device.clone(),
                ),
                Some(Tensor::randn::<f32>(
                    shape!(config.hidden_size),
                    device.clone(),
                )),
            ),
            norm: LayerNorm::new(
                Tensor::randn::<f32>(shape!(config.hidden_size), device.clone()),
                Some(Tensor::randn::<f32>(
                    shape!(config.hidden_size),
                    device.clone(),
                )),
                1e-5,
            ),
            n_heads: config.num_attention_heads as u32,
            attention_head_size: attention_head_size as u32,
            softmax_scale,
        };

        let mlp = MLP {
            up: Linear::new(
                Tensor::randn::<f32>(
                    shape!(config.intermediate_size, config.hidden_size),
                    device.clone(),
                ),
                Some(Tensor::randn::<f32>(
                    shape!(config.intermediate_size),
                    device.clone(),
                )),
            ),
            down: Linear::new(
                Tensor::randn::<f32>(
                    shape!(config.hidden_size, config.intermediate_size),
                    device.clone(),
                ),
                Some(Tensor::randn::<f32>(
                    shape!(config.hidden_size),
                    device.clone(),
                )),
            ),
        };
        Self {
            attn,
            mlp,
            norm: LayerNorm::new(
                Tensor::randn::<f32>(shape!(config.hidden_size), device.clone()),
                Some(Tensor::randn::<f32>(
                    shape!(config.hidden_size),
                    device.clone(),
                )),
                1e-5,
            ),
        }
    }
}
