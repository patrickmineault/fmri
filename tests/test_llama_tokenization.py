"""
Regression test for LLAMA tokenization functions.

This test verifies that the refactored LLAMA tokenization functions
produce the same results as the original functions.
"""

import torch
import numpy as np
import pytest
from ridge_utils.DataSequence import DataSequence
from ridge_utils.tokenization_helpers import (
    _compute_correct_tokens_llama,
    generate_efficient_feat_dicts,
    convert_to_feature_mats,
    LLAMA_BOS_TOKEN,
    LLAMA_WORD_BOUNDARY_TOKEN,
    LLAMA_SPACE_TOKEN,
)
from transformers import AutoTokenizer


def generate_efficient_feat_dicts_llama_old(wordseqs, tokenizer, lookback1, lookback2):
    """Original LLAMA implementation for comparison."""
    text_dict = {}
    text_dict2 = {}
    text_dict3 = {}
    
    for story in wordseqs.keys():
        ds = wordseqs[story]
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (LLAMA_WORD_BOUNDARY_TOKEN not in tokens)  # Use a dummy token for marking word cutoffs
        
        acc = [LLAMA_BOS_TOKEN]  # Contexts should start with special START token
        acc8 = 0
        acc_words = 0
        
        for ei, token_id in enumerate(tokens):
            token_str = tokenizer.convert_ids_to_tokens(torch.tensor([token_id]))[0]
            decoded_token = tokenizer.decode(torch.tensor([token_id]))
            
            if token_str.startswith('▁') and decoded_token.strip() != '':
                acc.append(LLAMA_WORD_BOUNDARY_TOKEN)
                acc.append(token_id)
                acc8 += 1
            elif ei != (len(tokens) - 1):
                if (token_id == LLAMA_SPACE_TOKEN and 
                    not tokenizer.convert_ids_to_tokens(torch.tensor([tokens[ei+1]]))[0].startswith('▁')):
                    acc.append(LLAMA_WORD_BOUNDARY_TOKEN)
                    acc.append(token_id)
                    acc8 += 1
                else:
                    acc.append(token_id)
            else:
                acc.append(token_id)
        
        # Validate word count
        acc_words = sum(1 for word in ds.data if word.strip() != '')
        assert acc8 == acc_words  # Number of annotations should equal number of words
        acc.append(LLAMA_WORD_BOUNDARY_TOKEN)
        
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [LLAMA_BOS_TOKEN]
        
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1 or (lookback2 > acc_lookback >= lookback1):
                    new_tokens = _compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif acc_lookback == lookback2:
                    new_tokens = _compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    text_dict3[tuple(new_tokens)] = False
                else:
                    assert False
            else:
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


@pytest.fixture(scope="session")
def test_data():
    """Create test data for LLAMA tokenization tests."""
    story_words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] * 20  # 180 words
    
    # Create random data times
    np.random.seed(42)  # For reproducible tests
    data_times = np.cumsum(np.random.rand(len(story_words)) + 0.5)
    
    # Create random TRs
    tr_times = np.arange(0, data_times[-1] + 10, 2)
    ds = DataSequence(story_words, [len(story_words)], data_times, tr_times)
    return ds


@pytest.fixture(scope="session")
def llama_tokenizer():
    """Load real LLAMA tokenizer for testing."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        return tokenizer
    except Exception as e:
        pytest.skip(f"Could not load LLAMA tokenizer: {e}")


@pytest.fixture
def wordseqs(test_data):
    """Create wordseqs dictionary for tests."""
    return {"test_story": test_data}


@pytest.fixture
def lookback_params():
    """Lookback parameters for tests."""
    return 10, 20  # Smaller values for faster testing


def test_llama_tokenization_constants():
    """Test that LLAMA constants are properly defined."""
    assert LLAMA_BOS_TOKEN == 1
    assert LLAMA_WORD_BOUNDARY_TOKEN == 29947
    assert LLAMA_SPACE_TOKEN == 29871


def test_llama_refactored_functions_keys(wordseqs, llama_tokenizer, lookback_params):
    """Test that refactored LLAMA functions return dictionaries with same keys as original."""
    lookback1, lookback2 = lookback_params
    
    # Test original functions
    text_dict_orig, text_dict2_orig, text_dict3_orig = generate_efficient_feat_dicts_llama_old(
        wordseqs, llama_tokenizer, lookback1, lookback2
    )
    
    # Test refactored functions
    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, llama_tokenizer, lookback1, lookback2
    )
    
    # Check that the dictionaries have the same keys
    assert set(text_dict_orig.keys()) == set(word_to_tokens.keys())
    assert set(text_dict2_orig.keys()) == set(compute_embeddings_flags.keys())
    assert set(text_dict3_orig.keys()) == set(token_sequence_registry.keys())


def test_llama_refactored_functions_values(wordseqs, llama_tokenizer, lookback_params):
    """Test that refactored LLAMA functions return same values as original."""
    lookback1, lookback2 = lookback_params
    
    # Test original functions
    text_dict_orig, text_dict2_orig, text_dict3_orig = generate_efficient_feat_dicts_llama_old(
        wordseqs, llama_tokenizer, lookback1, lookback2
    )
    
    # Test refactored functions
    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, llama_tokenizer, lookback1, lookback2
    )
    
    # Check that the values are the same
    for key in text_dict_orig.keys():
        assert text_dict_orig[key] == word_to_tokens[key], f"Token sequences don't match for key {key}"
        assert text_dict2_orig[key] == compute_embeddings_flags[key], f"Embedding flags don't match for key {key}"
    
    for key in text_dict3_orig.keys():
        assert text_dict3_orig[key] == token_sequence_registry[key], f"Token registry values don't match for key {key}"


def test_llama_feature_extraction_basic(wordseqs, llama_tokenizer, lookback_params):
    """Test that LLAMA feature extraction works with the refactored functions."""
    lookback1, lookback2 = lookback_params
    
    # Generate dictionaries
    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, llama_tokenizer, lookback1, lookback2
    )
    
    # Create mock embeddings
    embedding_dim = 128
    for token_sequence in token_sequence_registry:
        token_sequence_registry[token_sequence] = np.random.randn(embedding_dim)
    
    # Extract features
    feats = convert_to_feature_mats(wordseqs, word_to_tokens, token_sequence_registry)
    
    # Basic sanity checks
    assert len(feats) > 0
    for story_name, features in feats.items():
        assert features.shape[0] > 0  # Should have some time points
        assert features.shape[1] == embedding_dim  # Should have correct embedding dimension


def test_llama_tokenizer_detection(llama_tokenizer):
    """Test that LLAMA tokenizers are properly detected."""
    # Test with different LLAMA model names
    test_names = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf", 
        "huggingface/CodeLlama-7b-Python-hf",
        "Llama-2-7b-chat-hf"
    ]
    
    for name in test_names:
        # Create a copy of the tokenizer with different name
        test_tokenizer = llama_tokenizer
        test_tokenizer.name_or_path = name
        
        # This should not raise an exception
        try:
            wordseqs = {"test": DataSequence(["test", "words"], [2], [1.0, 2.0], [0, 2, 4])}
            generate_efficient_feat_dicts(wordseqs, test_tokenizer, 5, 10)
        except NotImplementedError:
            pytest.fail(f"LLAMA tokenizer {name} should be supported")


def test_compute_correct_tokens_llama():
    """Test the LLAMA token computation function directly."""
    # Create a simple annotated token sequence
    acc = [
        LLAMA_BOS_TOKEN,  # Start
        LLAMA_WORD_BOUNDARY_TOKEN, 278,  # "▁the"
        LLAMA_WORD_BOUNDARY_TOKEN, 4996,  # "▁quick" 
        LLAMA_WORD_BOUNDARY_TOKEN, 17354,  # "▁brown"
        LLAMA_WORD_BOUNDARY_TOKEN  # End marker
    ]
    
    # Test extracting tokens for different lookback windows
    tokens1 = _compute_correct_tokens_llama(acc, 0, 0, 3)  # First word only
    assert tokens1[0] == LLAMA_BOS_TOKEN
    assert 278 in tokens1  # Should contain "▁the"
    
    tokens2 = _compute_correct_tokens_llama(acc, 1, 1, 3)  # Two words
    assert tokens2[0] == LLAMA_BOS_TOKEN
    assert 278 in tokens2 and 4996 in tokens2  # Should contain "▁the" and "▁quick"


def test_llama_word_boundary_detection(llama_tokenizer):
    """Test that LLAMA word boundary detection works correctly."""
    # Test with a simple sentence
    test_words = ["the", "quick", "brown", "fox"]
    wordseqs = {"test": DataSequence(test_words, [len(test_words)], [1.0, 2.0, 3.0, 4.0], [0, 2, 4, 6, 8])}
    
    # Generate dictionaries
    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, llama_tokenizer, 2, 4
    )
    
    # Check that we have mappings for all words
    assert len(word_to_tokens) == len(test_words)
    
    # Check that all token sequences start with BOS token
    for tokens in word_to_tokens.values():
        assert tokens[0] == LLAMA_BOS_TOKEN


if __name__ == "__main__":
    pytest.main([__file__]) 