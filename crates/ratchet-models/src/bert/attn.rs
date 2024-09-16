
use ratchet::{Device, Tensor};
use ratchet_nn::{KVCache, LayerNorm, Linear, Module};

#[derive(Debug)]
pub struct BertSelfAttention{
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: u32,
    attention_head_size: u32, //  config.hidden_size / config.num_attention_heads
    softmax_scale: Tensor,
}

impl Module for BertSelfAttention{
    type Input= Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        todo!()
    }
}