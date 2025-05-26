"""Minimal example of extracting features from a Llama model.

This script loads an actual story from the repository, crops it to a maximum of
4000 words, and then uses :func:`generate_efficient_feat_dicts_llama` to create
the token lookback dictionaries.  The tiny random Llama model is used so the
example can run quickly.  The goal is to provide a short script that exercises
the feature extraction code path for debugging.
"""

import torch
from ridge_utils_frozen.ridge_utils.DataSequence import DataSequence
from ridge_utils_frozen.ridge_utils.tokenization_helpers import (
    generate_efficient_feat_dicts_llama,
    convert_to_feature_mats_llama,
)
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # Load one of the stories shipped with the repository and crop to 4000 words
    with open("data/wheretheressmoke_punc.txt", "r") as f:
        story_words = f.read().split()
    story_words = story_words[:4000]

    ds = DataSequence(story_words, [len(story_words)])
    wordseqs = {"wheretheressmoke": ds}

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    lookback1, lookback2 = 256, 512
    text_dict, text_dict2, text_dict3 = generate_efficient_feat_dicts_llama(
        wordseqs, tokenizer, lookback1, lookback2
    )

    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )

    layer_num = 1
    for phrase in text_dict2:
        if text_dict2[phrase]:
            inputs = {"input_ids": torch.tensor([text_dict[phrase]]).int()}
            inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape)
            out = list(model(**inputs, output_hidden_states=True)[2])
            out = out[layer_num][0].detach().numpy()
            this_key = tuple(inputs["input_ids"][0].numpy())
            for ei, _ in enumerate(this_key):
                prefix = this_key[: ei + 1]
                if prefix in text_dict3:
                    text_dict3[prefix] = out[ei, :]

    feats = convert_to_feature_mats_llama(
        wordseqs, tokenizer, lookback1, lookback2, text_dict3
    )

    for k, ds in feats.items():
        print(k, ds.data.shape)


if __name__ == "__main__":
    main()
