
use ratchet::{Device, Tensor};
use ratchet_nn::{KVCache, LayerNorm, Linear, Module};

use super::{attn::BertSelfAttention, embedding::BertEmbedding, layer::BertEncoderLayer, mlp::MLP};

pub struct BertInput {
    pub input_ids: Tensor,
    // pub attention_mask: Option<Vec<Vec<i32>>>,
}


#[derive(Debug)]
pub struct Bert{
    pub embedding: BertEmbedding,
    pub layers: Vec<BertEncoderLayer>, // num_hidden_layers   
    pub device: Device,
}

impl Module for Bert{
    type Input = BertInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let BertInput {
            input_ids,
        } = input;
        let [batch_size, seq_len]: [usize; 2] = input_ids.shape().try_into()?;
        println!("embedding sequence: batch_size: {}, seq_len: {}",batch_size,seq_len);
        
        let mut x = self.embedding.schedule(input_ids)?;
        for layer in &self.layers{
            x = layer.schedule(x)?;
        }
        Ok(x)
    }
}
