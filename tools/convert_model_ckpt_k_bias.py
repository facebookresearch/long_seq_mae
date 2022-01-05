# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import torch
from collections import OrderedDict


def convert_model_state_dict(state_dict, add_k_bias, remove_k_bias):
    state_dict = OrderedDict(state_dict)  # make a copy
    param_names = list(state_dict)
    for name in param_names:
        if name.endswith("attn.q_bias") and add_k_bias:
            name_q_bias = name
            name_v_bias = name.replace("attn.q_bias", "attn.v_bias")
            name_qkv_bias = name.replace("attn.q_bias", "attn.qkv.bias")
            q_bias = state_dict.pop(name_q_bias)
            k_bias = torch.zeros_like(q_bias)
            v_bias = state_dict.pop(name_v_bias)
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            state_dict[name_qkv_bias] = qkv_bias
            print(f"merge {name_q_bias} and {name_v_bias} into {name_qkv_bias}")

        if name.endswith("attn.qkv.bias") and remove_k_bias:
            name_qkv_bias = name
            name_q_bias = name.replace("attn.qkv.bias", "attn.q_bias")
            name_v_bias = name.replace("attn.qkv.bias", "attn.v_bias")
            qkv_bias = state_dict.pop(name_qkv_bias)
            assert qkv_bias.size(0) % 3 == 0
            dim = qkv_bias.size(0) // 3
            q_bias = state_dict[:dim]
            v_bias = state_dict[-dim:]
            state_dict[name_q_bias] = q_bias
            state_dict[name_v_bias] = v_bias
            print(f"split {name_qkv_bias} into {name_q_bias} and {name_v_bias}")

    return state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--add_k_bias", action="store_true")
    parser.add_argument("--remove_k_bias", action="store_true")
    args = parser.parse_args()

    assert args.add_k_bias != args.remove_k_bias, \
        "one and only one of --add_k_bias and --remove_k_bias should to be set"

    input_ckpt = torch.load(args.input, map_location="cpu")
    input_ckpt["model"] = convert_model_state_dict(
        input_ckpt["model"],
        args.add_k_bias,
        args.remove_k_bias,
    )
    torch.save(input_ckpt, args.output)
    print(f"saved output checkpoint to {args.output}")


if __name__ == "__main__":
    main()
