"""Minimal utilities for OPT tokenization used in tests."""

from typing import Dict, List, Tuple

from .DataSequence import DataSequence


def compute_correct_tokens_opt(
    acc: List[int], acc_lookback: int, acc_offset: int, total_len: int
) -> List[int]:
    """Return the token ids for the context ending at ``acc_offset``."""
    new_tokens: List[int] = [2]
    acc_count_all = 0
    first_word = max(0, acc_offset - acc_lookback)
    last_word = min(acc_offset + 1, total_len)
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


def generate_efficient_feat_dicts_opt(
    wordseqs: Dict[str, DataSequence], tokenizer, lookback1: int, lookback2: int
) -> Tuple[Dict[Tuple[str, int], List[int]], Dict[Tuple[str, int], bool], Dict[Tuple[int, ...], bool]]:
    """Tokenize each word once and compute look-back contexts."""
    tokens_by_word: Dict[Tuple[str, int], List[int]] = {}
    compute_flags: Dict[Tuple[str, int], bool] = {}
    embedding_cache: Dict[Tuple[int, ...], bool] = {}

    for story, ds in wordseqs.items():
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        tokens = tokenizer(text)["input_ids"][0]
        assert 27 not in tokens
        acc: List[int] = []
        for ei, tok in enumerate(tokens):
            dec = tokenizer.decode([tok])
            if (dec and dec[0] == " " and dec.strip() != "") or (
                dec != "</s>" and ei == 1
            ):
                acc.append(27)
                acc.append(tok)
            else:
                acc.append(tok)
        acc.append(27)

        acc_lookback = 0
        misc_offset = 0
        current_tokens = [2]
        for i, w in enumerate(ds.data):
            if w.strip() != "" and w != "'s":
                if acc_lookback < lookback1 or (
                    lookback2 > acc_lookback and acc_lookback >= lookback1
                ):
                    current_tokens = compute_correct_tokens_opt(
                        acc, acc_lookback, i + misc_offset, total_len
                    )
                    tokens_by_word[(story, i)] = current_tokens
                    compute_flags[(story, i)] = False
                    embedding_cache[tuple(current_tokens)] = False
                elif acc_lookback == lookback2:
                    current_tokens = compute_correct_tokens_opt(
                        acc, acc_lookback, i + misc_offset, total_len
                    )
                    acc_lookback = lookback1
                    tokens_by_word[(story, i)] = current_tokens
                    compute_flags[(story, i)] = True
                    embedding_cache[tuple(current_tokens)] = False
                else:
                    raise AssertionError("lookback edge case")
            else:
                tokens_by_word[(story, i)] = current_tokens
                compute_flags[(story, i)] = True
                embedding_cache[tuple(current_tokens)] = False
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                compute_flags[(story, i)] = True
    return tokens_by_word, compute_flags, embedding_cache


def convert_to_feature_mats_opt(
    wordseqs: Dict[str, DataSequence],
    tokens_by_word: Dict[Tuple[str, int], List[int]],
    embedding_cache: Dict[Tuple[int, ...], List[float]],
) -> Dict[str, DataSequence]:
    """Convert cached token embeddings into feature matrices."""
    featureseqs: Dict[str, DataSequence] = {}
    for story, ds in wordseqs.items():
        newdata = []
        for i in range(len(ds.data)):
            tok = tokens_by_word[(story, i)]
            newdata.append(embedding_cache[tuple(tok)])
        featureseqs[story] = DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)
    downsampled: Dict[str, DataSequence] = {}
    for story in featureseqs:
        downsampled[story] = featureseqs[story].chunksums("lanczos", window=3)
    return downsampled
