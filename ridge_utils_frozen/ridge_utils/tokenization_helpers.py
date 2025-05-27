import numpy as np
import logging
import sys
import joblib
import matplotlib.pyplot as plt
import torch
from ridge_utils.DataSequence import DataSequence
from transformers import AutoTokenizer, AutoModelForCausalLM


### Warning, you are entering tokenization hell.

# OPT tokenizer constants
OPT_BOS_TOKEN = 2
OPT_WORD_BOUNDARY_TOKEN = 27

# LLAMA tokenizer constants  
LLAMA_BOS_TOKEN = 1
LLAMA_WORD_BOUNDARY_TOKEN = 29947
LLAMA_SPACE_TOKEN = 29871

def _compute_correct_tokens_opt(acc, acc_lookback, acc_offset, total_len):
    #print(acc)
    new_tokens = []
    new_tokens.append(OPT_BOS_TOKEN) # Special OPT start token
    acc_count_all = 0
    first_word = max(0,acc_offset-acc_lookback)
    last_word = min(acc_offset+1, total_len)
    acc_start = 0
    while acc_start != first_word + 1:
        if acc[acc_count_all] == OPT_WORD_BOUNDARY_TOKEN:
            acc_start += 1
            acc_count_all += 1
        else:
            acc_count_all += 1
    
    acc2 = acc[acc_count_all:]
    acc_count8 = 0
    acc_count_all = 0
    while acc_count8 != (last_word - first_word):
        if acc2[acc_count_all] == OPT_WORD_BOUNDARY_TOKEN:
            acc_count8 += 1
            acc_count_all += 1
        else:
            new_tokens.append(acc2[acc_count_all])
            acc_count_all += 1
    return new_tokens


def _compute_correct_tokens_llama(acc, acc_lookback, acc_offset, total_len):
    new_tokens = [LLAMA_BOS_TOKEN]
    acc_count_all = 0
    first_word = max(0,acc_offset-acc_lookback)
    last_word = min(acc_offset+1, total_len)
    acc_start = 0
    while acc_start != first_word + 1:
        if acc[acc_count_all] == LLAMA_WORD_BOUNDARY_TOKEN:
            acc_start += 1
            acc_count_all += 1
        else:
            acc_count_all += 1
    acc2 = acc[acc_count_all:]
    acc_count8 = 0
    acc_count_all = 0
    while acc_count8 != (last_word - first_word):
        if acc2[acc_count_all] == LLAMA_WORD_BOUNDARY_TOKEN:
            acc_count8 += 1
            acc_count_all += 1
        else:
            new_tokens.append(acc2[acc_count_all])
            acc_count_all += 1
    return new_tokens


def _annotate_word_boundaries_opt(words, tokenizer, story):
    """
    Tokenize words and annotate word boundaries with special marker (27).
    
    Args:
        words: List of words in the story
        tokenizer: OPT tokenizer
        story: Story name for edge case handling
        
    Returns:
        List of tokens with word boundary markers (27) inserted
    """
    text = [" ".join(words)]
    inputs = tokenizer(text, return_tensors="pt")
    tokens = np.array(inputs['input_ids'][0])
    assert (OPT_WORD_BOUNDARY_TOKEN not in tokens)  # Ensure our boundary marker isn't in the original tokens
    
    annotated_tokens = []
    
    for ei, token_id in enumerate(tokens):
        # Handle various tokenization edge cases for word boundaries
        decoded_token = tokenizer.decode(torch.tensor([token_id]))
        
        if ((decoded_token[0] == ' ' and decoded_token.strip() != '') or 
            (decoded_token != '</s>' and ei == 1)):
            annotated_tokens.append(OPT_WORD_BOUNDARY_TOKEN)  # Word boundary marker
            annotated_tokens.append(token_id)
        elif _is_edge_case_word_boundary_opt(ei, token_id, story):
            annotated_tokens.append(OPT_WORD_BOUNDARY_TOKEN)  # Word boundary marker
            annotated_tokens.append(token_id)
        else:
            annotated_tokens.append(token_id)
    
    annotated_tokens.append(OPT_WORD_BOUNDARY_TOKEN)  # Final boundary marker
    return annotated_tokens


def _annotate_word_boundaries_llama(words, tokenizer, story):
    """
    Tokenize words and annotate word boundaries with special marker for LLAMA.
    
    Args:
        words: List of words in the story
        tokenizer: LLAMA tokenizer
        story: Story name (unused for LLAMA but kept for consistency)
        
    Returns:
        List of tokens with word boundary markers inserted
    """
    text = [" ".join(words)]
    inputs = tokenizer(text, return_tensors="pt")
    tokens = np.array(inputs['input_ids'][0])
    assert (LLAMA_WORD_BOUNDARY_TOKEN not in tokens)  # Ensure our boundary marker isn't in the original tokens
    
    annotated_tokens = [LLAMA_BOS_TOKEN]  # Start with BOS token
    
    for ei, token_id in enumerate(tokens):
        # LLAMA uses ▁ (sentencepiece underscore) to mark word beginnings
        token_str = tokenizer.convert_ids_to_tokens(torch.tensor([token_id]))[0]
        decoded_token = tokenizer.decode(torch.tensor([token_id]))
        
        if token_str.startswith('▁') and decoded_token.strip() != '':
            annotated_tokens.append(LLAMA_WORD_BOUNDARY_TOKEN)
            annotated_tokens.append(token_id)
        elif ei != (len(tokens) - 1):
            # Special case for space tokens followed by non-underscore tokens
            if (token_id == LLAMA_SPACE_TOKEN and 
                not tokenizer.convert_ids_to_tokens(torch.tensor([tokens[ei+1]]))[0].startswith('▁')):
                annotated_tokens.append(LLAMA_WORD_BOUNDARY_TOKEN)
                annotated_tokens.append(token_id)
            else:
                annotated_tokens.append(token_id)
        else:
            annotated_tokens.append(token_id)
    
    annotated_tokens.append(LLAMA_WORD_BOUNDARY_TOKEN)  # Final boundary marker
    return annotated_tokens


def _is_edge_case_word_boundary_opt(token_idx, token_id, story):
    """Handle specific tokenization edge cases for word boundaries."""
    edge_cases = [
        (1860, 2836), (349, 1437), (365, 1437), (1914, 1437), (1305, 1437),
        (202, 3432), (1316, 4514), (656, 2550), (1358, 6355), (2160, 8629)
    ]
    
    # Special case for specific story
    if token_idx == 300 and token_id == 1437 and story == 'beneaththemushroomcloud':
        return True
    
    # General edge cases
    if (token_idx, token_id) in edge_cases:
        return True
        
    # Special token case
    if token_id == 24929 and token_idx != OPT_BOS_TOKEN:
        return True
        
    return False


# Legacy functions for backward compatibility
def compute_correct_tokens_llama(acc, acc_lookback, acc_offset, total_len):
    """Legacy function - use _compute_correct_tokens_llama instead."""
    return _compute_correct_tokens_llama(acc, acc_lookback, acc_offset, total_len)


def generate_efficient_feat_dicts_llama(wordseqs, tokenizer, lookback1, lookback2):
    """Legacy function - use generate_efficient_feat_dicts instead."""
    return generate_efficient_feat_dicts(wordseqs, tokenizer, lookback1, lookback2)


def convert_to_feature_mats_llama(wordseqs, tokenizer, lookback1, lookback2, text_dict3):
    """Legacy function - use convert_to_feature_mats instead."""
    # Convert old-style text_dict3 to new-style embedding_cache
    word_to_tokens, _, _ = generate_efficient_feat_dicts(wordseqs, tokenizer, lookback1, lookback2)
    return convert_to_feature_mats(wordseqs, word_to_tokens, text_dict3)


def generate_efficient_feat_dicts(wordseqs, tokenizer, half_window, full_window):
    """
    Generate efficient feature dictionaries.

    Args:
        wordseqs: Dict mapping story name to DataSequence
        tokenizer: Tokenizer (supports facebook/opt-125m and LLAMA models)
        half_window: Half window size
        full_window: Full window size
    
    Returns:
        word_to_tokens: Dict mapping (story, word_index) to list of token IDs
        compute_embeddings_flags: Dict mapping (story, word_index) to bool indicating if embeddings should be computed
        token_sequence_registry: Dict mapping tuple of token IDs to placeholder for embeddings
    """
    word_to_tokens = {}
    compute_embeddings_flags = {}
    token_sequence_registry = {}

    # Determine tokenizer type and set appropriate functions
    if "opt" in tokenizer.name_or_path.lower():
        BOS_TOKEN = OPT_BOS_TOKEN
        WORD_BOUNDARY_TOKEN = OPT_WORD_BOUNDARY_TOKEN
        annotate_word_boundaries = _annotate_word_boundaries_opt
        compute_correct_tokens = _compute_correct_tokens_opt
    elif "llama" in tokenizer.name_or_path.lower() or "Llama" in tokenizer.name_or_path:
        BOS_TOKEN = LLAMA_BOS_TOKEN
        WORD_BOUNDARY_TOKEN = LLAMA_WORD_BOUNDARY_TOKEN
        annotate_word_boundaries = _annotate_word_boundaries_llama
        compute_correct_tokens = _compute_correct_tokens_llama
    else:
        raise NotImplementedError(f"Tokenizer {tokenizer.name_or_path} not supported")
    
    for story in wordseqs.keys():
        ds = wordseqs[story]
        total_len = len(ds.data)
        
        # Tokenize the entire story and annotate word boundaries
        annotated_tokens = annotate_word_boundaries(ds.data, tokenizer, story)
        
        # Validate word count for LLAMA (similar to original implementation)
        if "llama" in tokenizer.name_or_path.lower() or "Llama" in tokenizer.name_or_path:
            word_count = sum(1 for word in ds.data if word.strip() != '')
            boundary_count = annotated_tokens.count(WORD_BOUNDARY_TOKEN) - 1  # Subtract final boundary
            assert boundary_count == word_count, f"Word count mismatch: {boundary_count} boundaries vs {word_count} words"
        
        # Process each word with lookback logic
        lookback_count = 0
        misc_offset = 0
        current_tokens = [BOS_TOKEN]
        
        for word_idx, word in enumerate(ds.data):
            if word.strip() != '' and word != "'s":
                # Generate tokens for current context window
                if lookback_count < half_window:
                    current_tokens = compute_correct_tokens(annotated_tokens, lookback_count, word_idx + misc_offset, total_len)
                    word_to_tokens[(story, word_idx)] = current_tokens
                    compute_embeddings_flags[(story, word_idx)] = False
                    token_sequence_registry[tuple(current_tokens)] = False
                elif full_window > lookback_count >= half_window:
                    current_tokens = compute_correct_tokens(annotated_tokens, lookback_count, word_idx + misc_offset, total_len)
                    word_to_tokens[(story, word_idx)] = current_tokens
                    compute_embeddings_flags[(story, word_idx)] = False
                    token_sequence_registry[tuple(current_tokens)] = False
                elif lookback_count == full_window:
                    current_tokens = compute_correct_tokens(annotated_tokens, lookback_count, word_idx + misc_offset, total_len)
                    lookback_count = half_window  # Reset lookback
                    word_to_tokens[(story, word_idx)] = current_tokens
                    compute_embeddings_flags[(story, word_idx)] = True
                    token_sequence_registry[tuple(current_tokens)] = False
                else:
                    print("WARNING, LOOKBACK EDGE CASE 1", lookback_count, "\n")
                    assert False
            else:
                # Handle empty words or contractions
                word_to_tokens[(story, word_idx)] = current_tokens
                compute_embeddings_flags[(story, word_idx)] = True
                token_sequence_registry[tuple(current_tokens)] = False
                lookback_count += 1
                misc_offset -= 1
                continue
                
            lookback_count += 1
            if word_idx == total_len - 1:
                compute_embeddings_flags[(story, word_idx)] = True
                
    return word_to_tokens, compute_embeddings_flags, token_sequence_registry

def convert_to_feature_mats(wordseqs, word_to_tokens, embedding_cache):
    """
    Convert word sequences to feature matrices using pre-computed embeddings.
    
    Args:
        wordseqs: Dictionary of story name to DataSequence
        word_to_tokens: Dictionary mapping (story, word_index) to token sequences
        embedding_cache: Dictionary mapping token sequences to embeddings
        
    Returns:
        Dictionary of story name to downsampled feature matrices
    """
    featureseqs = {}
    
    for story in wordseqs.keys():
        ds = wordseqs[story]
        word_features = []
        
        for word_idx, word in enumerate(ds.data):
            # Get the token sequence for this word
            token_sequence = tuple(word_to_tokens[(story, word_idx)])
            
            # Look up the embedding in the cache
            if token_sequence in embedding_cache and isinstance(embedding_cache[token_sequence], np.ndarray):
                word_features.append(embedding_cache[token_sequence])
            else:
                # This shouldn't happen if the pipeline is set up correctly
                raise ValueError(f"No embedding found for word {word_idx} in story {story}")
        
        featureseqs[story] = DataSequence(np.array(word_features), ds.split_inds, ds.data_times, ds.tr_times)
    
    # Downsample features
    downsampled_featureseqs = {}
    for story in featureseqs:
        downsampled_featureseqs[story] = featureseqs[story].chunksums('lanczos', window=3)
    
    return downsampled_featureseqs