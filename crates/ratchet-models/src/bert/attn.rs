use ratchet::{shape, Shape, Tensor};
use ratchet_nn::{LayerNorm, Linear, Module};

#[derive(Debug)]
pub struct BertSelfAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub norm: LayerNorm,
    pub n_heads: u32,
    pub attention_head_size: u32, //  config.hidden_size / config.num_attention_heads
    pub softmax_scale: Tensor,
}

impl Module for BertSelfAttention {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let [batch_size, seq_len, emb_dim]: [usize; 3] = input.shape().try_into()?;
        let residual = input.clone();

        let transpose_shape: Shape = [
            batch_size,
            seq_len,
            self.n_heads as usize,
            self.attention_head_size as usize,
        ]
        .as_slice()
        .into();

        let q_state = self.q.schedule(input.clone())?;
        let k_state = self.k.schedule(input.clone())?;
        let v_state = self.v.schedule(input.clone())?;

        let q_state = q_state
            .view(transpose_shape.clone())?
            .permute([0, 2, 1, 3].as_slice())?;
        let k_state = k_state
            .view(transpose_shape.clone())?
            .permute([0, 2, 1, 3].as_slice())?;
        let v_state = v_state
            .view(transpose_shape.clone())?
            .permute([0, 2, 1, 3].as_slice())?;

        let attn_weights = q_state
            .full()?
            .matmul(k_state.permute(&[0, 1, 3, 2])?.full()?, false, false)?
            .mul(self.softmax_scale.clone())?;

        let w = attn_weights.softmax(3)?.cast(v_state.dt())?;
        let wv = w.matmul(v_state, false, false)?.permute(&[0, 2, 1, 3])?;
        let wv = wv.view(shape![batch_size as _, seq_len, emb_dim])?;
        let output = self.out.schedule(wv)?;
        self.norm.schedule(output.add(residual)?)
    }
}
