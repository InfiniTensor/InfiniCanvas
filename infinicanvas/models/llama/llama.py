import numpy as np
from ...modeling import DTYPE, InfiniTensorModel
from ...transformer import (
    DecoderLayer,
    RMSNorm,
    InfiniTensorTokenizer,
)
from ...nn import Linear


class LlamaConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pretraining_tp=1,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        dtype=DTYPE.F32,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.dtype = dtype

        self.batch_size = "batch_size"
        self.seq_len = "seq_len"


class LlamaModel(InfiniTensorModel):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.embed_tokens = self.parameter(
            (config.vocab_size, config.hidden_size),
            config.dtype,
            "embed_tokens",
        )
        self.decoders = [
            self.make_submodel(DecoderLayer, config, model_name=f"Decoder_{i}")
            for i in range(self.num_layers)
        ]
        self.layernorm = self.make_submodel(
            RMSNorm,
            config.hidden_size,
            config.rms_norm_eps,
            config.dtype,
            model_name="layernorm",
        )
        self.lm_head = self.make_submodel(
            Linear,
            config.hidden_size,
            config.vocab_size,
            False,
            config.dtype,
            model_name="lm_head",
        )

    def forward(
        self, token_ids, pos_ids, attention_mask=None
    ):
        hidden_states = self.gather(self.embed_tokens, token_ids, axis=0)
        for i in range(self.num_layers):
            hidden_states = self.decoders[i](
                hidden_states, pos_ids, attention_mask
            )
        hidden_states = self.layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, input_text: str, tokenizer: InfiniTensorTokenizer, top_k=3, top_p=1.0, temperature=1.0, max_length = 100):
        token_ids, pos_ids = tuple(self.inputs[:2])
        attention_mask = self.inputs[2] if len(self.inputs) >=3 else None

        input_tokens = tokenizer.encode(input_text)
        batch_size, seq_len, past_seq_len = 1, len(input_tokens), 0
        input_data = np.array(input_tokens).reshape((batch_size, seq_len))
        inputs = {token_ids: input_data}
        if attention_mask is not None:
            inputs[attention_mask] = np.triu(
                np.ones((seq_len, seq_len), dtype=self.config.dtype.np_type())
                * -np.inf,
                1,
            )
        inputs[pos_ids] = np.tile(np.arange(seq_len, dtype=np.int64) + past_seq_len,(
            (batch_size, 1))
        )

        variable_map = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "past_seq_len": past_seq_len,
            "total_seq_len": past_seq_len + seq_len,
        }

        output = self.run(inputs, variable_map)[0]

        whole_text = ""
        whole_text += input_text
        print(input_text, end="")
        output = output[:, -1, None, :]
        output_token = random_search(output, top_k, top_p, temperature)
        output_text = tokenizer.decode(output_token.flatten().tolist())
        print(output_text, end="")
        whole_text += output_text

        max_length = min(max_length + seq_len, self.config.hidden_size)
        for pos_id in range(seq_len, max_length):
            inputs[token_ids] = output_token
            inputs[pos_ids] = np.full((batch_size, 1), pos_id)
            if attention_mask is not None:
                inputs[attention_mask] = np.zeros(
                    (batch_size, 1, 1, pos_id + 1), dtype=self.config.dtype.np_type()
                )
            variable_map["seq_len"] = 1
            variable_map["past_seq_len"] = pos_id
            variable_map["total_seq_len"] = (
                variable_map["seq_len"] + variable_map["past_seq_len"]
            )
            output = self.run(inputs, variable_map)[0]
            output_token = random_search(output, top_k, top_p, temperature)
            output_text = tokenizer.decode(output_token.flatten().tolist())
            print(output_text, end="")
            whole_text += output_text
            if whole_text.endswith("</s>"):
                break
        print("")
        return whole_text

    def load_param_hf(self, model_path):
        pass

    def load_param_safetensors(self, model_path):
        from safetensors import safe_open
        import os

        tensors = {}
        with safe_open(
            os.path.join(model_path, "model.safetensors"), framework="np", device="cpu"
        ) as f:
            for k in f.keys():
                data = f.get_tensor(k)
                naming = k.split(".")
                tensor_name = ""
                if naming[0] == "lm_head":
                    tensor_name = "LlamaModel/lm_head/weight"
                elif naming[0] == "model" and naming[1] == "embed_tokens":
                    tensor_name = "LlamaModel/embed_tokens"
                elif naming[0] == "model" and naming[1] == "norm":
                    tensor_name = "LlamaModel/layernorm/weight"
                elif naming[0] == "model" and naming[1] == "layers":
                    tensor_name = f"LlamaModel/Decoder_{naming[2]}/"
                    if naming[3] == "self_attn":
                        tensor_name += f"Attention/{naming[4]}/weight"
                    elif naming[3] == "mlp":
                        tensor_name += f"FeedForward/{naming[4]}/weight"
                    else:
                        tensor_name += f"{naming[3]}/weight"
                else:
                    print(f"Unmatched param: {k}")
                tensors[tensor_name] = data
        self.load_params(tensors)


def greedy_search(logits) -> np.ndarray:
    assert len(logits.shape) == 3
    return np.argmax(logits, axis=-1)


def random_search(logits, top_k=5, top_p=1.0, temperature=1.0) -> np.ndarray:
    if temperature == 0:
        return greedy_search(logits)
    assert len(logits.shape) == 3
    bs, seq_l, vocab_size = logits.shape

    logits = logits.reshape(bs * seq_l, vocab_size) / temperature
    logits = logits - logits.max()
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    sorted_indices = np.flip(np.argsort(probabilities, axis=-1)[:, -top_k:], axis=-1)
    cumulative_prob = np.cumsum(probabilities[np.arange(bs * seq_l)[:, None], sorted_indices], axis=-1)
    mask = cumulative_prob <= top_p

    result = []
    for i in range(bs * seq_l):
        filtered_indices = sorted_indices[i, mask[i]]
        filtered_prob = probabilities[i][filtered_indices]
        filtered_prob = filtered_prob - filtered_prob.max()
        filtered_prob = np.exp(filtered_prob) / np.sum(np.exp(filtered_prob))
        result.append(np.random.choice(filtered_indices, p=filtered_prob))

    return np.array(result).reshape(bs, seq_l)

# Deprecated
# class ROPE_HF:
#     def __init__(self, dim, dtype, max_position_embeddings=4096, base=10000) -> None:
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2) / self.dim))
#         self.inv_freq = inv_freq
#         self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=dtype)

#     def _set_cos_sin_cache(self, seq_len, dtype):
#         self.max_seq_len_cached = seq_len
#         t = np.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

#         freqs = np.einsum("i,j->ij", t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = np.concatenate((freqs, freqs), axis=-1)
#         self.cos_cached = np.cos(emb).astype(dtype)
#         self.sin_cached = np.sin(emb).astype(dtype)

#     def __call__(self, pos_ids):
#         return self.cos_cached[pos_ids].copy(), self.sin_cached[pos_ids].copy()
