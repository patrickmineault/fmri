#!/usr/bin/env python3
"""
Test script for Gemma 2 tokenization support in ridge_utils.

This script tests the tokenization helpers with real Gemma 2 models to ensure
proper word boundary detection and token processing.
"""

import sys
import os
import numpy as np
import torch
import pytest
from transformers import AutoTokenizer

from ridge_utils.tokenization_helpers import (
    generate_efficient_feat_dicts,
    convert_to_feature_mats,
    GEMMA2_BOS_TOKEN,
    GEMMA2_WORD_BOUNDARY_TOKEN,
    _annotate_word_boundaries_gemma2,
    _compute_correct_tokens_gemma2
)
from ridge_utils.DataSequence import DataSequence


@pytest.fixture(scope="session")
def gemma2_tokenizer():
    """Load real Gemma 2 tokenizer for testing."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        return tokenizer
    except Exception as e:
        pytest.skip(f"Could not load Gemma 2 tokenizer: {e}")


@pytest.fixture
def sample_wordseqs():
    # Load test data
    try:
        with open("../data/wheretheressmoke_punc.txt", "r") as f:
            story_words = f.read().split()
    except FileNotFoundError:
        # Fallback to dummy data if file not found
        raise "Test data not found"
    
    story_words = story_words[:211]  # Use smaller subset for faster testing
    print(story_words[-10:])

    # create random data times
    np.random.seed(42)  # For reproducible tests
    data_times = np.cumsum(np.random.rand(len(story_words)) + .5)

    # Create random TRs
    tr_times = np.arange(0, data_times[-1] + 10, 2)
    ds = DataSequence(story_words, [len(story_words)], data_times, tr_times)

    return {
        "test_story": ds
    }

def test_gemma2_tokenizer_detection(gemma2_tokenizer, sample_wordseqs):
    """Test that Gemma 2 tokenizers are properly detected."""
    # Test various Gemma 2 model names by modifying the tokenizer's name_or_path
    gemma2_names = [
        "google/gemma-2-9b",
        "google/gemma-2-27b", 
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "google/gemma-2-2b",
        "Google/Gemma-2-9B",  # Test case sensitivity
    ]
    
    for name in gemma2_names:
        # Create a copy of the tokenizer with different name
        test_tokenizer = gemma2_tokenizer
        original_name = test_tokenizer.name_or_path
        test_tokenizer.name_or_path = name
        
        try:
            # This should not raise an exception for Gemma 2 models
            word_to_tokens, compute_flags, token_registry = generate_efficient_feat_dicts(
                sample_wordseqs, test_tokenizer, half_window=128, full_window=256
            )
            assert len(word_to_tokens) > 0, f"No word-to-token mappings generated for {name}"
            assert len(compute_flags) > 0, f"No compute flags generated for {name}"
            assert len(token_registry) > 0, f"No token registry generated for {name}"
        except NotImplementedError:
            pytest.fail(f"Gemma 2 tokenizer {name} should be supported")
        finally:
            # Restore original name
            test_tokenizer.name_or_path = original_name


def test_gemma2_word_boundary_annotation(gemma2_tokenizer):
    """Test word boundary annotation for Gemma 2."""
    words = ["Hello", "world", "test"]
    story = "test_story"
    
    annotated_tokens = _annotate_word_boundaries_gemma2(words, gemma2_tokenizer, story)
    
    # Check that we have the BOS token at the start
    assert annotated_tokens[0] == GEMMA2_BOS_TOKEN, f"Expected BOS token {GEMMA2_BOS_TOKEN} at start"
    
    # Check that we have word boundary markers
    assert GEMMA2_WORD_BOUNDARY_TOKEN in annotated_tokens, "Expected word boundary markers in annotated tokens"
    
    # Check that the final token is a word boundary marker
    assert annotated_tokens[-1] == GEMMA2_WORD_BOUNDARY_TOKEN, "Expected final word boundary marker"
    
    # Check that the sequence is reasonable length
    assert len(annotated_tokens) > len(words), "Annotated tokens should be longer than input words"


def test_gemma2_token_computation(gemma2_tokenizer):
    """Test token computation for Gemma 2."""
    words = ["Hello", "world", "test"]
    story = "test_story"
    
    # Get real annotated tokens
    annotated_tokens = _annotate_word_boundaries_gemma2(words, gemma2_tokenizer, story)
    
    # Test different lookback scenarios
    test_cases = [
        (0, 0, len(words), "First word, no lookback"),
        (1, 1, len(words), "Second word, lookback=1"),
        (2, 2, len(words), "Third word, lookback=2"),
    ]
    
    for acc_lookback, acc_offset, total_len, description in test_cases:
        tokens = _compute_correct_tokens_gemma2(
            annotated_tokens, acc_lookback, acc_offset, total_len
        )
        
        # Check that we always start with BOS token
        assert tokens[0] == GEMMA2_BOS_TOKEN, f"Expected BOS token {GEMMA2_BOS_TOKEN} at start"
        
        # Check that we don't have boundary tokens in the result (except BOS)
        boundary_tokens_in_result = [t for t in tokens[1:] if t == GEMMA2_WORD_BOUNDARY_TOKEN]
        assert len(boundary_tokens_in_result) == 0, "Boundary tokens should be filtered out from result"
        
        # Check that result is not empty
        assert len(tokens) > 1, f"Token sequence should have more than just BOS token for {description}"


def test_gemma2_integration(gemma2_tokenizer, sample_wordseqs):
    """Test full integration with generate_efficient_feat_dicts."""
    word_to_tokens, compute_flags, token_registry = generate_efficient_feat_dicts(
        sample_wordseqs, gemma2_tokenizer, half_window=128, full_window=256
    )
    
    assert len(word_to_tokens) > 0, "Should generate word-to-token mappings"
    assert len(compute_flags) > 0, "Should generate compute flags"
    assert len(token_registry) > 0, "Should generate token sequences"
    
    # Check that we have mappings for all words
    story_name = "test_story"
    test_words = sample_wordseqs[story_name].data
    
    for i, word in enumerate(test_words):
        key = (story_name, i)
        assert key in word_to_tokens, f"Missing word-to-token mapping for word {i}: {word}"
        assert key in compute_flags, f"Missing compute flag for word {i}: {word}"
        
        tokens = word_to_tokens[key]
        assert tokens[0] == GEMMA2_BOS_TOKEN, f"Expected BOS token at start of sequence for word {i}"
        assert len(tokens) > 1, f"Token sequence should have more than just BOS token for word {i}"


def test_gemma2_real_tokenization(gemma2_tokenizer):
    """Test that real Gemma 2 tokenization works as expected."""
    test_text = "The quick brown fox jumps"
    
    # Test basic tokenization
    tokens = gemma2_tokenizer([test_text])
    token_ids = tokens['input_ids'][0]
    
    assert len(token_ids) > 0, "Should produce token IDs"
    assert token_ids[0] == GEMMA2_BOS_TOKEN, f"First token should be BOS token {GEMMA2_BOS_TOKEN}"
    
    # Test token-to-string conversion
    token_strings = gemma2_tokenizer.convert_ids_to_tokens(token_ids)
    assert len(token_strings) == len(token_ids), "Should have same number of token strings as IDs"
    
    # Test that SentencePiece-style tokens are present (tokens starting with ▁)
    underscore_tokens = [t for t in token_strings if isinstance(t, str) and t.startswith('▁')]
    assert len(underscore_tokens) > 0, "Should have SentencePiece-style tokens with ▁ prefix"
    
    # Test decoding
    decoded = gemma2_tokenizer.decode(token_ids, skip_special_tokens=True)
    assert isinstance(decoded, str), "Decoded text should be a string"
    assert len(decoded.strip()) > 0, "Decoded text should not be empty"


def test_gemma2_feature_matrix_conversion(gemma2_tokenizer, sample_wordseqs):
    """Test conversion to feature matrices."""
    word_to_tokens, compute_flags, token_registry = generate_efficient_feat_dicts(
        sample_wordseqs, gemma2_tokenizer, half_window=128, full_window=256
    )
    
    # Create mock embeddings for each unique token sequence
    embedding_dim = 256  # Use smaller dimension for testing
    mock_embeddings = {}
    
    for token_seq in token_registry.keys():
        mock_embedding = np.random.randn(embedding_dim)
        mock_embeddings[token_seq] = mock_embedding
    
    # Convert to feature matrices
    feature_matrices = convert_to_feature_mats(sample_wordseqs, word_to_tokens, mock_embeddings)
    
    assert len(feature_matrices) > 0, "Should generate feature matrices"
    
    for story_name, feature_seq in feature_matrices.items():
        assert story_name in sample_wordseqs, f"Story {story_name} should be in original wordseqs"
        
        if isinstance(feature_seq, DataSequence):
            # It's a DataSequence object
            assert feature_seq.data is not None, "Feature matrix data should not be None"
            assert feature_seq.data.shape[1] == embedding_dim, f"Feature matrix should have {embedding_dim} columns"
            assert feature_seq.data.shape[0] > 0, "Feature matrix should have rows"
        else:
            # It's a numpy array (from chunksums)
            assert feature_seq.shape[1] == embedding_dim, f"Feature matrix should have {embedding_dim} columns"
            assert feature_seq.shape[0] > 0, "Feature matrix should have rows"


def test_gemma2_edge_cases(gemma2_tokenizer):
    """Test edge cases for Gemma 2 tokenization."""
    # Test empty word list
    empty_words = []
    story = "empty_story"
    
    # This should handle empty input gracefully
    try:
        annotated_tokens = _annotate_word_boundaries_gemma2(empty_words, gemma2_tokenizer, story)
        # Should at least have BOS and final boundary marker
        assert len(annotated_tokens) >= 2, "Should have at least BOS and boundary marker for empty input"
        assert annotated_tokens[0] == GEMMA2_BOS_TOKEN, "Should start with BOS token"
        assert annotated_tokens[-1] == GEMMA2_WORD_BOUNDARY_TOKEN, "Should end with boundary marker"
    except Exception as e:
        pytest.fail(f"Empty word list should be handled gracefully: {e}")
    
    # Test single word
    single_word = ["Hello"]
    annotated_tokens = _annotate_word_boundaries_gemma2(single_word, gemma2_tokenizer, story)
    assert len(annotated_tokens) >= 3, "Should have BOS, word tokens, and boundary marker"
    assert annotated_tokens[0] == GEMMA2_BOS_TOKEN, "Should start with BOS token"
    assert annotated_tokens[-1] == GEMMA2_WORD_BOUNDARY_TOKEN, "Should end with boundary marker"
    
    # Test words with special characters
    special_words = ["Hello!", "world?", "test."]
    annotated_tokens = _annotate_word_boundaries_gemma2(special_words, gemma2_tokenizer, story)
    assert len(annotated_tokens) > len(special_words), "Should handle special characters"
    assert annotated_tokens[0] == GEMMA2_BOS_TOKEN, "Should start with BOS token"
    assert annotated_tokens[-1] == GEMMA2_WORD_BOUNDARY_TOKEN, "Should end with boundary marker"


if __name__ == "__main__":
    pytest.main([__file__]) 