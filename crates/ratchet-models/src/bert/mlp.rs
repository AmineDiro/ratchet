use ratchet::Tensor;
use ratchet_nn::{Linear, Module};

#[derive(Debug)]
pub struct MLP{
    up: Linear, // hidden_size, intermediate_size
    down: Linear //  intermediate_size, hidden_size
}

impl Module for MLP{
    type Input= Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<ratchet::Tensor> {
        self.down.schedule(self.up.schedule(input)?.gelu()?)
        
    }
}