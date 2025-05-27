"""
Regression test for refactored tokenization functions.

This test verifies that the refactored versions of generate_efficient_feat_dicts_opt
and convert_to_feature_mats_opt produce the same results as the original functions.
"""

import torch
import numpy as np
import pytest
from ridge_utils.DataSequence import DataSequence
from ridge_utils.tokenization_helpers import (
    _compute_correct_tokens_opt,
    generate_efficient_feat_dicts,
    convert_to_feature_mats,
)
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_efficient_feat_dicts_opt_old(wordseqs, tokenizer, lookback1, lookback2):
    text_dict = {}
    text_dict2 = {}
    text_dict3 = {}
    for story in wordseqs.keys():
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        acc = []
        acc8 = 0
        text = [" ".join(ds.data)]
        text_len = len(text[0])
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (27 not in tokens)
        # Annotate word boundaries
        for ei,i in enumerate(tokens):
            # A lot of tokenization edge cases
            if (tokenizer.decode(torch.tensor([i]))[0] == ' ' and tokenizer.decode(torch.tensor([i])).strip() != '') or (tokenizer.decode(torch.tensor([i])) != '</s>' and ei == 1):
                acc.append(27)
                acc.append(i)
                acc8 += 1
            elif (ei==1860 and i == 2836) or (ei==349 and i == 1437) or (ei==365 and i == 1437) or (ei==1914 and i == 1437) or (ei==1305 and i == 1437) or (ei==300 and i==1437 and story=='beneaththemushroomcloud') or (ei==202 and i == 3432) or (ei==1316 and i==4514) or (ei==656 and i==2550) or (ei==1358 and i==6355) or (ei==2160 and i==8629) or (i==24929 and ei != 2):
                acc.append(27)
                acc.append(i)
                acc8 += 1
            else:
                acc.append(i)
        acc.append(27)
        #print(acc)
        lookback1 = 256
        lookback2 = 512
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [2]
        #print(tokenizer.decode(new_tokens))
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1:
                    new_tokens = _compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    #print(tokenizer.decode(torch.tensor(new_tokens)))
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif lookback2 > acc_lookback and acc_lookback >= lookback1:
                    new_tokens = _compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    #print(tokenizer.decode(torch.tensor(new_tokens)))
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif acc_lookback == lookback2:
                    new_tokens = _compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    #print(tokenizer.decode(torch.tensor(new_tokens)))
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    text_dict3[tuple(new_tokens)] = False
                else:
                    print("WARNING, LOOKBACK EDGE CASE 1", acc_lookback, "\n")
                    assert False
                    #print(max(0, i-acc_lookback), min(i+1, total_len))
                    #text = [" ".join(ds.data[max(0,i-acc_lookback):min(i+1,total_len)])][0]
                    #print(text)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
            else:
                #hidden_states = np.zeros((1024,))
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                text_dict3[tuple(new_tokens)] = False
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True
    return text_dict, text_dict2, text_dict3


def convert_to_feature_mats_opt_old(wordseqs, tokenizer, lookback1, lookback2, text_dict3):
    text_dict = {}
    text_dict2 = {}
    featureseqs = {}
    for story in wordseqs.keys():
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        acc = []
        acc8 = 0
        text = [" ".join(ds.data)]
        text_len = len(text[0])
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (27 not in tokens)
        # Annotate word boundaries
        for ei,i in enumerate(tokens):
            # A lot of tokenization edge cases
            if (tokenizer.decode(torch.tensor([i]))[0] == ' ' and tokenizer.decode(torch.tensor([i])).strip() != '') or (tokenizer.decode(torch.tensor([i])) != '</s>' and ei == 1):
                acc.append(27)
                acc.append(i)
                acc8 += 1
            elif (ei==1860 and i == 2836) or (ei==349 and i == 1437) or (ei==365 and i == 1437) or (ei==1914 and i == 1437) or (ei==1305 and i == 1437) or (ei==300 and i==1437 and story=='beneaththemushroomcloud') or (ei==202 and i == 3432) or (ei==1316 and i==4514) or (ei==656 and i==2550) or (ei==1358 and i==6355) or (ei==2160 and i==8629) or (i==24929 and ei != 2):
                acc.append(27)
                acc.append(i)
                acc8 += 1
            else:
                acc.append(i)
        acc.append(27)
        lookback1 = 256
        lookback2 = 512
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [2]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1:
                    new_tokens = _compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif lookback2 > acc_lookback and acc_lookback >= lookback1:
                    new_tokens = _compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif acc_lookback == lookback2:
                    new_tokens = _compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    newdata.append(text_dict3[tuple(new_tokens)])
                else:
                    print("WARNING, LOOKBACK EDGE CASE 1", acc_lookback, "\n")
                    assert False
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
            else:
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                newdata.append(text_dict3[tuple(new_tokens)])
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True
        featureseqs[story] = DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)
    downsampled_featureseqs = {}
    for story in featureseqs:
        downsampled_featureseqs[story] = featureseqs[story].chunksums('lanczos', window=3)
    return downsampled_featureseqs

@pytest.fixture(scope="session")
def ds():
    # Load test data
    try:
        with open("../data/wheretheressmoke_punc.txt", "r") as f:
            story_words = f.read().split()
    except FileNotFoundError:
        # Fallback to dummy data if file not found
        story_words = ["the", "quick", "brown", "fox", "jumps"] * 400
    
    story_words = story_words[:2000]  # Use smaller subset for faster testing

    # create random data times
    np.random.seed(42)  # For reproducible tests
    data_times = np.cumsum(np.random.rand(len(story_words)) + .5)

    # Create random TRs
    tr_times = np.arange(0, data_times[-1] + 10, 2)
    ds = DataSequence(story_words, [len(story_words)], data_times, tr_times)
    return ds

def test_refactored_functions(ds):
    """Test that refactored functions produce the same results as original functions."""

    wordseqs = {"wheretheressmoke": ds}
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    lookback1, lookback2 = 256, 512
    
    # Test original functions
    text_dict_orig, text_dict2_orig, text_dict3_orig = generate_efficient_feat_dicts_opt_old(
        wordseqs, tokenizer, lookback1, lookback2
    )
    
    # Test refactored functions
    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, tokenizer, lookback1, lookback2
    )
    
    # Check that the dictionaries have the same keys
    assert set(text_dict_orig.keys()) == set(word_to_tokens.keys())
    assert set(text_dict2_orig.keys()) == set(compute_embeddings_flags.keys())
    assert set(text_dict3_orig.keys()) == set(token_sequence_registry.keys())
    
    # Check that the values are the same
    for key in text_dict_orig.keys():
        assert text_dict_orig[key] == word_to_tokens[key], f"Token sequences don't match for key {key}"
        assert text_dict2_orig[key] == compute_embeddings_flags[key], f"Embedding flags don't match for key {key}"
    
    for key in text_dict3_orig.keys():
        assert text_dict3_orig[key] == token_sequence_registry[key], f"Token registry values don't match for key {key}"

def test_feature_extraction_refactored(ds):
    """Test that refactored feature extraction produces same results as original."""
    
    wordseqs = {"wheretheressmoke": ds}
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    lookback1, lookback2 = 256, 512
    
    # Generate dictionaries using both methods
    text_dict_orig, text_dict2_orig, text_dict3_orig = generate_efficient_feat_dicts_opt_old(
        wordseqs, tokenizer, lookback1, lookback2
    )
    
    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, tokenizer, lookback1, lookback2
    )
    
    # Create mock embeddings for testing
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    layer_num = 1
    
    # Populate embeddings in both dictionaries
    for phrase in text_dict2_orig:
        if text_dict2_orig[phrase]:
            inputs = {"input_ids": torch.tensor([text_dict_orig[phrase]]).int()}
            inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape)
            out = list(model(**inputs, output_hidden_states=True)[2])
            out = out[layer_num][0].detach().numpy()
            this_key = tuple(inputs["input_ids"][0].numpy())
            for ei, _ in enumerate(this_key):
                prefix = this_key[: ei + 1]
                if prefix in text_dict3_orig:
                    text_dict3_orig[prefix] = out[ei, :]

    for word, compute_embedding in compute_embeddings_flags.items():
        if compute_embedding:
            inputs = {"input_ids": torch.tensor([word_to_tokens[word]]).int()}
            inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape)
            out = list(model(**inputs, output_hidden_states=True)[2])
            out = out[layer_num][0].detach().numpy()
            this_key = tuple(inputs["input_ids"][0].numpy())
            for ei, _ in enumerate(this_key):
                prefix = this_key[: ei + 1]
                if prefix in token_sequence_registry:
                    token_sequence_registry[prefix] = out[ei, :]
    
    # Test feature extraction
    feats_orig = convert_to_feature_mats_opt_old(
        wordseqs, tokenizer, lookback1, lookback2, text_dict3_orig
    )
    
    feats_refactored = convert_to_feature_mats(
        wordseqs, word_to_tokens, token_sequence_registry
    )
    
    # Compare feature extraction results
    assert set(feats_orig.keys()) == set(feats_refactored.keys())
    
    for story in feats_orig.keys():
        orig_shape = feats_orig[story].shape
        refactored_shape = feats_refactored[story].shape
        assert orig_shape == refactored_shape, f"Feature shapes don't match for {story}: {orig_shape} vs {refactored_shape}"
        assert orig_shape[0] * orig_shape[1] > 10000, "Not enough features"
        
        # Check if the features are close (allowing for small numerical differences)
        np.testing.assert_allclose(
            feats_orig[story], 
            feats_refactored[story], 
            rtol=1e-5, 
            atol=1e-8,
            err_msg=f"Features don't match for {story}"
        )

def test_unified_interface_opt(ds):
    """Test that the unified interface works correctly for OPT tokenization."""
    wordseqs = {"wheretheressmoke": ds}
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    lookback1, lookback2 = 256, 512
    
    # Test that the unified interface produces the same results as the old OPT-specific functions
    text_dict_orig, text_dict2_orig, text_dict3_orig = generate_efficient_feat_dicts_opt_old(
        wordseqs, tokenizer, lookback1, lookback2
    )
    
    # Use the unified interface
    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, tokenizer, lookback1, lookback2
    )
    
    # Results should be identical
    assert set(text_dict_orig.keys()) == set(word_to_tokens.keys())
    assert set(text_dict2_orig.keys()) == set(compute_embeddings_flags.keys())
    assert set(text_dict3_orig.keys()) == set(token_sequence_registry.keys())
    
    for key in text_dict_orig.keys():
        assert text_dict_orig[key] == word_to_tokens[key]
        assert text_dict2_orig[key] == compute_embeddings_flags[key]
    
    for key in text_dict3_orig.keys():
        assert text_dict3_orig[key] == token_sequence_registry[key]

def test_tokenizer_detection():
    """Test that different tokenizer types are properly detected."""
    # Test OPT detection
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    # This should work without raising an exception
    wordseqs = {"test": DataSequence(["test", "words"], [2], [1.0, 2.0], [0, 2, 4])}
    word_to_tokens, _, _ = generate_efficient_feat_dicts(wordseqs, tokenizer, 5, 10)
    assert len(word_to_tokens) > 0
    
    # Test unsupported tokenizer
    class UnsupportedTokenizer:
        def __init__(self):
            self.name_or_path = "unsupported/model"
    
    unsupported_tokenizer = UnsupportedTokenizer()
    
    with pytest.raises(NotImplementedError, match="not supported"):
        generate_efficient_feat_dicts(wordseqs, unsupported_tokenizer, 5, 10)

