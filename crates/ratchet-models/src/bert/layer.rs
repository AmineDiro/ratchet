use ratchet::Tensor;
use ratchet_nn::Module;

use super::{attn::BertSelfAttention,  mlp::MLP};

#[derive(Debug)]
pub struct BertEncoderLayer{
    attn : BertSelfAttention,
    mlp: MLP,
    norm: ratchet_nn::LayerNorm
}
impl Module for BertEncoderLayer{
    type Input= Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<ratchet::Tensor> {
        let mut x = self.attn.schedule(input)?;
        let residual = x.clone();
        x = self.mlp.schedule(x)?;
        x = self.norm.schedule(x.add(residual)?)?;
        Ok(x)
    }
}