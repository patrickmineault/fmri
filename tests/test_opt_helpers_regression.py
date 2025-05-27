import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
pytest.importorskip("numpy")

from ridge_utils_frozen.ridge_utils import tokenization_helpers as new_helpers

# Dummy tokenizer replicating minimal OPT behaviour
class DummyTokenizer:
    def __init__(self):
        self.token_map = {}
        self.id_map = {}
        self.next_id = 3
        self.start = 2
    def __call__(self, texts):
        text = texts[0]
        ids = [self.start]
        for word in text.split():
            tok = " " + word
            if tok not in self.token_map:
                self.token_map[tok] = self.next_id
                self.id_map[self.next_id] = tok
                self.next_id += 1
            ids.append(self.token_map[tok])
        return {"input_ids": [ids]}
    def decode(self, ids):
        if isinstance(ids, list):
            if len(ids) == 1:
                return self.id_map.get(ids[0], "")
            return "".join(self.id_map[i] for i in ids)
        return self.id_map.get(ids, "")

# Copies of the original helper functions prior to refactor

def old_generate_efficient_feat_dicts_opt(wordseqs, tokenizer, lookback1, lookback2):
    text_dict = {}
    text_dict2 = {}
    text_dict3 = {}
    for story in wordseqs:
        ds = wordseqs[story]
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        tokens = tokenizer(text)["input_ids"][0]
        acc = []
        for ei, i in enumerate(tokens):
            dec = tokenizer.decode([i])
            if (dec and dec[0] == " " and dec.strip() != "") or (dec != "</s>" and ei == 1):
                acc.append(27)
                acc.append(i)
            else:
                acc.append(i)
        acc.append(27)
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [2]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1:
                    new_tokens = new_helpers.compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif lookback2 > acc_lookback and acc_lookback >= lookback1:
                    new_tokens = new_helpers.compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                elif acc_lookback == lookback2:
                    new_tokens = new_helpers.compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    text_dict3[tuple(new_tokens)] = False
                else:
                    raise AssertionError("lookback edge case")
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


def old_convert_to_feature_mats_opt(wordseqs, tokenizer, lookback1, lookback2, text_dict3):
    text_dict = {}
    text_dict2 = {}
    featureseqs = {}
    for story in wordseqs:
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        tokens = tokenizer(text)["input_ids"][0]
        acc = []
        for ei, i in enumerate(tokens):
            dec = tokenizer.decode([i])
            if (dec and dec[0] == " " and dec.strip() != "") or (dec != "</s>" and ei == 1):
                acc.append(27)
                acc.append(i)
            else:
                acc.append(i)
        acc.append(27)
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [2]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1:
                    new_tokens = new_helpers.compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif lookback2 > acc_lookback and acc_lookback >= lookback1:
                    new_tokens = new_helpers.compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif acc_lookback == lookback2:
                    new_tokens = new_helpers.compute_correct_tokens_opt(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    newdata.append(text_dict3[tuple(new_tokens)])
                else:
                    raise AssertionError("lookback edge case")
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
        featureseqs[story] = new_helpers.DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)
    downsampled_featureseqs = {}
    for story in featureseqs:
        downsampled_featureseqs[story] = featureseqs[story].chunksums('lanczos', window=3)
    return downsampled_featureseqs


def test_refactored_matches_old():
    words = ["hello", "world", "this", "is", "a", "test"]
    ds = new_helpers.DataSequence(words, [len(words)])
    wordseqs = {"s": ds}
    tokenizer = DummyTokenizer()
    lookback1, lookback2 = 2, 4

    old_td, old_td2, old_td3 = old_generate_efficient_feat_dicts_opt(wordseqs, tokenizer, lookback1, lookback2)
    new_td, new_td2, new_td3 = new_helpers.generate_efficient_feat_dicts_opt(wordseqs, tokenizer, lookback1, lookback2)

    assert old_td == new_td
    assert old_td2 == new_td2
    assert old_td3 == new_td3

    for k in old_td3:
        old_td3[k] = f"v{k}"
        new_td3[k] = f"v{k}"

    old_feats = old_convert_to_feature_mats_opt(wordseqs, tokenizer, lookback1, lookback2, old_td3)
    new_feats = new_helpers.convert_to_feature_mats_opt(wordseqs, new_td, new_td3)

    assert list(old_feats['s'].data) == list(new_feats['s'].data)
