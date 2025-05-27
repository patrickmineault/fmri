import numpy as np
import logging
import sys
import joblib
import matplotlib.pyplot as plt
import torch
from ridge_utils.DataSequence import DataSequence
from transformers import AutoTokenizer, AutoModelForCausalLM


### Warning, you are entering tokenization hell.

def compute_correct_tokens_opt(acc, acc_lookback, acc_offset, total_len):
    #print(acc)
    new_tokens = []
    new_tokens.append(2) # Special OPT start token
    acc_count_all = 0
    first_word = max(0,acc_offset-acc_lookback)
    last_word = min(acc_offset+1, total_len)
    acc_start = 0
    while acc_start != first_word + 1:
        if acc[acc_count_all] == 27:
            acc_start += 1
            acc_count_all += 1
        else:
            acc_count_all += 1
    
    acc2 = acc[acc_count_all:]
    acc_count8 = 0
    acc_count_all = 0
    while acc_count8 != (last_word - first_word):
        if acc2[acc_count_all] == 27:
            acc_count8 += 1
            acc_count_all += 1
        else:
            new_tokens.append(acc2[acc_count_all])
            acc_count_all += 1
    return new_tokens


def compute_correct_tokens_llama(acc, acc_lookback, acc_offset, total_len):
    new_tokens = [1]
    acc_count_all = 0
    first_word = max(0,acc_offset-acc_lookback)
    last_word = min(acc_offset+1, total_len)
    acc_start = 0
    while acc_start != first_word + 1:
        if acc[acc_count_all] == 29947:
            acc_start += 1
            acc_count_all += 1
        else:
            acc_count_all += 1
    acc2 = acc[acc_count_all:]
    acc_count8 = 0
    acc_count_all = 0
    while acc_count8 != (last_word - first_word):
        if acc2[acc_count_all] == 29947:
            acc_count8 += 1
            acc_count_all += 1
        else:
            new_tokens.append(acc2[acc_count_all])
            acc_count_all += 1
    return new_tokens

def generate_efficient_feat_dicts_llama(wordseqs, tokenizer, lookback1, lookback2):
    text_dict = {}
    text_dict2 = {}
    text_dict3 = {}
    for es, story in enumerate(wordseqs.keys()):
        #print(story)
        ds = wordseqs[story]
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (29947 not in tokens) # Use a dummy token '8' for marking word cutoffs
        acc = [1] # Contexts should start with special START token
        acc8 = 0
        acc_words = 0
        for ei,i in enumerate(tokens):
            if tokenizer.convert_ids_to_tokens(torch.tensor([i]))[0][0] == '▁'  and (tokenizer.decode(torch.tensor([i])).strip() != ''):
                acc.append(29947)
                acc.append(i)
                acc8 += 1
            elif ei != (len(tokens) - 1):
                if (i == 29871) and (tokenizer.convert_ids_to_tokens(torch.tensor([tokens[ei+1]]))[0][0] != '▁'):
                    acc.append(29947)
                    acc.append(i)
                    acc8 += 1
                else:
                    acc.append(i)
            else:
                acc.append(i)
        decoded = tokenizer.decode(torch.tensor(acc))
        acc_words = 0
        for i in ds.data:
            if i.strip() != '':
                acc_words += 1
        #print(acc8, acc_words, story, es)
        assert acc8 == acc_words # Number of annotations should equal number of words
        acc.append(29947)
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [1]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1 or (lookback2 > acc_lookback and acc_lookback >= lookback1):
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
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

def convert_to_feature_mats_llama(wordseqs, tokenizer, lookback1, lookback2, text_dict3):
    text_dict = {}
    text_dict2 = {}
    featureseqs = {}
    for es, story in enumerate(wordseqs.keys()):
        #print(story)
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (29947 not in tokens) # Use a dummy token '8' for marking word cutoffs
        acc = [1] # Contexts should start with special START token
        acc8 = 0
        acc_words = 0
        for ei,i in enumerate(tokens):
            if tokenizer.convert_ids_to_tokens(torch.tensor([i]))[0][0] == '▁'  and (tokenizer.decode(torch.tensor([i])).strip() != ''):
                acc.append(29947)
                acc.append(i)
                acc8 += 1
            elif ei != (len(tokens) - 1):
                if (i == 29871) and (tokenizer.convert_ids_to_tokens(torch.tensor([tokens[ei+1]]))[0][0] != '▁'):
                    acc.append(29947)
                    acc.append(i)
                    acc8 += 1
                else:
                    acc.append(i)
            else:
                acc.append(i)
        decoded = tokenizer.decode(torch.tensor(acc))
        acc_words = 0
        for i in ds.data:
            if i.strip() != '':
                acc_words += 1
        #print(acc8, acc_words, story, es)
        assert acc8 == acc_words # Number of annotations should equal number of words
        acc.append(29947)
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [1]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1 or (lookback2 > acc_lookback and acc_lookback >= lookback1):
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    newdata.append(text_dict3[tuple(new_tokens)])
                else:
                    assert False
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

OPT_BOS_TOKEN = 2
OPT_WORD_BOUNDARY_TOKEN = 27

def generate_efficient_feat_dicts_opt(wordseqs, tokenizer, lookback1, lookback2):
    """
    Generate efficient feature dictionaries for OPT models.
    
    Returns:
        word_to_tokens: Dict mapping (story, word_index) to list of token IDs
        compute_embeddings_flags: Dict mapping (story, word_index) to bool indicating if embeddings should be computed
        token_sequence_registry: Dict mapping tuple of token IDs to placeholder for embeddings
    """
    word_to_tokens = {}
    compute_embeddings_flags = {}
    token_sequence_registry = {}
    
    for story in wordseqs.keys():
        ds = wordseqs[story]
        total_len = len(ds.data)
        
        # Tokenize the entire story and annotate word boundaries
        annotated_tokens = _annotate_word_boundaries_opt(ds.data, tokenizer, story)
        
        # Process each word with lookback logic
        lookback_count = 0
        misc_offset = 0
        current_tokens = [OPT_BOS_TOKEN]  # OPT start token
        
        for word_idx, word in enumerate(ds.data):
            if word.strip() != '' and word != "'s":
                # Generate tokens for current context window
                if lookback_count < lookback1:
                    current_tokens = compute_correct_tokens_opt(annotated_tokens, lookback_count, word_idx + misc_offset, total_len)
                    word_to_tokens[(story, word_idx)] = current_tokens
                    compute_embeddings_flags[(story, word_idx)] = False
                    token_sequence_registry[tuple(current_tokens)] = False
                elif lookback2 > lookback_count >= lookback1:
                    current_tokens = compute_correct_tokens_opt(annotated_tokens, lookback_count, word_idx + misc_offset, total_len)
                    word_to_tokens[(story, word_idx)] = current_tokens
                    compute_embeddings_flags[(story, word_idx)] = False
                    token_sequence_registry[tuple(current_tokens)] = False
                elif lookback_count == lookback2:
                    current_tokens = compute_correct_tokens_opt(annotated_tokens, lookback_count, word_idx + misc_offset, total_len)
                    lookback_count = lookback1  # Reset lookback
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

def convert_to_feature_mats_opt(wordseqs, word_to_tokens, embedding_cache):
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