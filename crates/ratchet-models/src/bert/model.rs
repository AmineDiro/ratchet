use ratchet::{Device, Tensor};
use ratchet_nn::Module;

use super::{embedding::BertEmbedding, layer::BertEncoderLayer};

pub struct BertInput {
    pub input_ids: Tensor,
    // pub attention_mask: Option<Vec<Vec<i32>>>,
}
pub struct BertConfig {
    pub vocab_size: usize,                 // Vocabulary size of the BERT model
    pub hidden_size: usize, // Dimensionality of the encoder layers and the pooler layer
    pub num_hidden_layers: usize, // Number of hidden layers in the Transformer encoder
    pub num_attention_heads: usize, // Number of attention heads for each attention layer
    pub intermediate_size: usize, // Dimensionality of the intermediate layer in the Transformer encoder
    pub hidden_act: String, // The non-linear activation function used in the encoder and pooler
    pub hidden_dropout_prob: f32, // Dropout probability for all fully connected layers
    pub attention_probs_dropout_prob: f32, // Dropout ratio for attention probabilities
    pub max_position_embeddings: usize, // Maximum sequence length for this model
    pub type_vocab_size: usize, // Vocabulary size for `token_type_ids`
    pub initializer_range: f32, // Standard deviation for weight initialization
    pub layer_norm_eps: f64, // Epsilon used by layer normalization layers
    pub position_embedding_type: String, // Type of position embedding: "absolute", "relative_key", or "relative_key_query"
    pub is_decoder: bool,                // Whether the model is used as a decoder or not
    pub use_cache: bool, // Whether the model should return the last key/values attentions
    pub classifier_dropout: Option<f32>, // Optional dropout ratio for the classification head
}

impl Default for BertConfig {
    fn default() -> Self {
        BertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: String::from("gelu"),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            position_embedding_type: String::from("absolute"),
            is_decoder: false,
            use_cache: true,
            classifier_dropout: None,
        }
    }
}

#[derive(Debug)]
pub struct Bert {
    pub embedding: BertEmbedding,
    pub layers: Vec<BertEncoderLayer>, // num_hidden_layers
    pub device: Device,
}

impl Module for Bert {
    type Input = BertInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let BertInput { input_ids } = input;
        let [batch_size, seq_len]: [usize; 2] = input_ids.shape().try_into()?;
        println!(
            "embedding sequence of shape: batch_size: {}, seq_len: {}",
            batch_size, seq_len
        );

        let mut x = self.embedding.schedule(input_ids)?;
        for layer in &self.layers {
            x = layer.schedule(x)?;
        }
        Ok(x)
    }
}

impl Bert {
    pub fn new(config: BertConfig, device: Device) -> Self {
        let embedding = BertEmbedding::new(&config, device.clone());
        let layers = (0..config.num_hidden_layers)
            .map(|_| BertEncoderLayer::new(&config, device.clone()))
            .collect();

        Self {
            embedding,
            layers,
            device,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bert::model::BertConfig;

    use super::{Bert, BertInput};
    use hf_hub::api::sync::Api;
    use ratchet::{prelude::shape, Device, DeviceRequest, Tensor};
    use ratchet_nn::Module;
    use tokenizers::Tokenizer;

    #[test]
    pub fn fake_embed_bert() -> anyhow::Result<()> {
        let device = Device::request_device(DeviceRequest::GPU)?;
        let config = BertConfig::default();
        let input = BertInput {
            input_ids: Tensor::randint(
                1i32,
                config.vocab_size as i32,
                shape!(1, 512),
                device.clone(),
            ),
        };
        dbg!(&input.input_ids.dt());
        let model = Bert::new(config, device.clone());
        let hidden_state = model.schedule(input)?.resolve()?.to(&Device::CPU)?;
        dbg!(hidden_state.shape());
        Ok(())
    }

    #[test]
    fn load_gte_small() -> anyhow::Result<()> {
        let api = Api::new().unwrap();
        let tokenizer_repo = api.model("microsoft/phi-2".to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let input_str = "this is test";
        let encoding = tokenizer.encode(input_str, true).unwrap();
        let tokens = encoding
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        Ok(())
    }
}
