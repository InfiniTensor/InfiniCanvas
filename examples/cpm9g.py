import json
from typing import (
    Any,
    Dict,
    List,
    IO,
)
import re
import io
import os
import glob
from pytrie import StringTrie
from infinicanvas.transformer.tokenizer import InfiniTensorTokenizer


def load_vocab(fp: IO[bytes]) -> Dict[str, int]:
    """Loads a vocabulary file into a dictionary."""
    vocab: Dict[str, int] = {}

    reader = io.TextIOWrapper(fp, encoding="utf-8")
    for token in reader.readlines():
        token = token.strip()
        if len(token) == 0:
            continue
        token = json.loads(token)
        vocab[token] = len(vocab)
    return vocab


class CPM9GTokenizer(InfiniTensorTokenizer):
    def __init__(self, path):
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.byte_list = ["<0x0{}>".format(hex(i).upper()[2:]) for i in range(0x10)] + [
            "<0x{}>".format(hex(i).upper()[2:]) for i in range(0x10, 0x100)
        ]

        self._special_token_set = set(
            [self.unk_token, self.bos_token, self.eos_token] + self.byte_list
        )

        all_tokens = load_vocab(io.FileIO(path, "rb"))

        self.encoder: Dict[str, int] = {}
        self._special_encoder: Dict[str, int] = {}
        for token, token_id in all_tokens.items():
            if token in self._special_token_set:
                self._special_encoder[token] = token_id
            else:
                self.encoder[token] = token_id

        self.decoder = {v: k for k, v in self.encoder.items()}
        self._byte_decoder = {
            self._special_encoder[token]: i for i, token in enumerate(self.byte_list)
        }

        self._max_word_len = max([len(x) for x in self.encoder.keys()])

        self._len_word_first = {}
        for x in self.encoder.keys():
            if not x[0] in self._len_word_first:
                self._len_word_first[x[0]] = 1
            if len(x) > self._len_word_first[x[0]]:
                self._len_word_first[x[0]] = len(x)
        self.tencoder = StringTrie(self.encoder)

    def get_piece(self, text: str) -> str:
        if text[0] in self._len_word_first:
            text = text[: self._len_word_first[text[0]]]
            len_text = len(text)
            for i in range(len(text)):
                sub = text[: len_text - i]
                if sub in self.encoder:
                    return sub
        return text[0]

    @property
    def vocab_size(self):
        return len(self)

    @property
    def eos_id(self):
        return self._special_encoder[self.eos_token]

    @property
    def bos_id(self):
        return self._special_encoder[self.bos_token]

    @property
    def unk_id(self):
        return self._special_encoder[self.unk_token]

    def __len__(self):
        return len(self.encoder) + len(self._special_encoder)

    def tokenize(self, text: str) -> List[str]:
        output_tokens: List[str] = []
        st = 0
        while st < len(text):
            piece = self.get_piece(text[st:])
            output_tokens.append(piece)
            st += len(piece)
        return output_tokens

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text

    def encode(self, text: str, with_bos=True) -> List[int]:
        ret = []
        if with_bos:
            ret.append(self.bos_id)
        for x in self.tokenize(text):
            if x in self.encoder:
                ret.append(self.encoder[x])
            else:
                ret.extend(self._encode_unicode(x))
        return ret

    def decode(self, tokens: List[int]):
        """Decode ids into a string."""
        ret = []
        st = 0

        while st < len(tokens):
            if tokens[st] in self.decoder:
                ret.append(self.decoder[tokens[st]])
                st += 1
            elif tokens[st] in self._byte_decoder:
                first = self._byte_decoder[tokens[st]]
                length = (
                    1 if first < 128 else len(re.search("^1+0", bin(first)[2:])[0]) - 1
                )
                code = 0
                try:
                    for j in range(length):
                        code = code << 8 | self._byte_decoder[tokens[st + j]]
                    code = int.to_bytes(code, length, "big").decode("utf-8")
                    ret.append(code)
                except:
                    pass
                st = st + length
            elif tokens[st] == self.eos_id:
                ret.append(self.eos_token)
                st += 1
            elif tokens[st] == self.bos_id:
                ret.append(self.bos_token)
                st += 1
            else:
                ret.append(self.unk_token)
                st += 1
        return "".join(ret)

    def _encode_unicode(self, token):
        # wrap unicode encoding into a helper function
        ids = []
        utf8_id = token.encode("utf-8")
        for _id in utf8_id:
            ids.append(self._special_encoder[self.byte_list[_id]])
        return ids

    def next_token(self, text):
        # fast next token matching
        token, token_id = self.tencoder.longest_prefix_item(text, (None, None))
        if token is None:
            token = text[0]
            token_ids = self._encode_unicode(token)
        else:
            token_ids = [token_id]
        return token, token_ids


def load_model_pt(model_dir):
    import torch
    import numpy as np

    state_dict = {}
    if os.path.isfile(f"{model_dir}"):
        state_dict = torch.load(f"{model_dir}", map_location="cpu")
    elif os.path.isfile(f"{model_dir}/pytorch_model.pt"):
        state_dict = torch.load(f"{model_dir}/pytorch_model.pt", map_location="cpu")
    else:
        pt_files = sorted(glob.glob(f"{model_dir}/pytorch_model*.bin"))
        if not pt_files:
            raise ValueError(f"No checkpoint found in: {model_dir}")
        for f in pt_files:
            state_dict.update(torch.load(f, map_location="cpu"))

    tensors = {}
    for name, data in state_dict.items():
        naming = name.split(".")
        tensor_name = ""
        if naming[0] == "lm_head":
            tensor_name = "LlamaModel/lm_head/weight"
        elif naming[0] == "input_embedding":
            tensor_name = "LlamaModel/embed_tokens"
        elif naming[0] == "encoder" and naming[1] == "output_layernorm":
            tensor_name = "LlamaModel/layernorm/weight"
        elif naming[0] == "encoder" and naming[1] == "layers":
            tensor_name = f"LlamaModel/Decoder_{naming[2]}/"
            if naming[3] == "self_att":
                if naming[4] == "self_attention":
                    if naming[5] == "project_q":
                        tensor_name += "Attention/q_proj/weight"
                    elif naming[5] == "project_k":
                        tensor_name += "Attention/k_proj/weight"
                    elif naming[5] == "project_v":
                        tensor_name += "Attention/v_proj/weight"
                    elif naming[5] == "attention_out":
                        tensor_name += "Attention/o_proj/weight"
                elif naming[4] == "layernorm_before_attention":
                    tensor_name += "input_layernorm/weight"
            elif naming[3] == "ffn":
                if naming[4] == "layernorm_before_ffn":
                    tensor_name += "post_attention_layernorm/weight"
                else:
                    if naming[5] == "w_in" and naming[6] == "w_0":
                        tensor_name += "FeedForward/gate_proj/weight"
                    elif naming[5] == "w_in" and naming[6] == "w_1":
                        tensor_name += "FeedForward/up_proj/weight"
                    elif naming[5] == "w_out":
                        tensor_name += "FeedForward/down_proj/weight"
        else:
            print(f"Unmatched param: {name}")

        tensors[tensor_name] = data.numpy()

    return tensors


def main(model_path):
    from infinicanvas.models.llama import LlamaModel, LlamaConfig
    from infinicanvas.modeling import DTYPE

    settings = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "max_position_embeddings": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 48,
        "num_key_value_heads": 32,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "use_cache": True,
        "vocab_size": 119696,
        "dtype": DTYPE.FP16,
    }

    tokenizer = CPM9GTokenizer(os.path.join(model_path, "vocabs.txt"))
    config = LlamaConfig(**settings)
    model = LlamaModel(config)
    inputs = ["token_ids", "r_embedding_cos", "r_embedding_sin", "attention_mask"]
    model(*inputs)
    model.to("nvidia", 9)
    model.load_params(load_model_pt(os.path.join(model_path, "cpm9g-11b-sft.pt")))

    model.generate(
        "Once upon a time,",
        tokenizer,
    )


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1]
    main(model_path)
