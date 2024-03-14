from ..modeling import InfiniTensorModel
from ..transformer import Attention, FeedForward, RMSNorm


class DecoderLayer(InfiniTensorModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attention_layer = self.make_submodel(
            Attention,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            dtype=config.dtype,
            attention_bias=config.attention_bias,
            use_kv_cache=config.use_cache,
            rope_theta = config.rope_theta,
            rope_scaling = config.rope_scaling,
        )
        self.input_layernorm = self.make_submodel(
            RMSNorm,
            config.hidden_size,
            config.rms_norm_eps,
            config.dtype,
            model_name="input_layernorm",
        )
        self.post_attention_layernorm = self.make_submodel(
            RMSNorm,
            config.hidden_size,
            config.rms_norm_eps,
            config.dtype,
            model_name="post_attention_layernorm",
        )
        self.feed_forward_layer = self.make_submodel(
            FeedForward, config.hidden_size, config.intermediate_size, config.dtype
        )

    def forward(
        self, hidden_states, pos_ids, attention_mask=None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention_layer(
            hidden_states, pos_ids, attention_mask
        )
        hidden_states = self.add(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward_layer(hidden_states)
        hidden_states = self.add(hidden_states, residual)

        return hidden_states
