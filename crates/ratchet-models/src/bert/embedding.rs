use ratchet::{Tensor};
use ratchet_nn::{Embedding, LayerNorm, Module};

#[derive(Debug)]
pub struct BertEmbedding{
    word_embedding: Embedding,// vocab_size, hidden_size
    positional_embedding: Embedding, // type_vocab_size, hidden_size
    // token_type_embedding: Embedding,  // max_positional_embedding, hidden_size
    norm: LayerNorm,
}

impl Module for BertEmbedding{
    type Input= Tensor;

    fn schedule(&self, input_ids: Self::Input) -> anyhow::Result<Tensor> {
        let [batch_size, seq_len]: [usize;2] = input_ids.shape().try_into()?;

        let input_embedding = self.word_embedding.schedule(input_ids)?;

        let position_ids = (0..seq_len as i32).collect::<Vec<_>>();
        let position_ids = Tensor::from_data(position_ids, [batch_size,seq_len].as_slice().into(),input_embedding.device().clone());

        let pos_embeddings = self.positional_embedding.schedule(position_ids)?;

        let mut embedding =  input_embedding.add(pos_embeddings)?;

        embedding = self.norm.schedule(embedding)?;

        Ok(embedding)
    }
}