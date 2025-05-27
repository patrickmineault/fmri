import joblib
from tqdm import tqdm
from ridge_utils.dsutils import make_word_ds
from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts, convert_to_feature_mats
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    # These files are located in the story_data folder of the Box
    grids = joblib.load("../data/grids_huge.jbl") # Load TextGrids containing story annotations
    trfiles = joblib.load("../data/trfiles_huge.jbl") # Load TRFiles containing TR information

    # We'll build an encoding model using this set of stories for this tutorial.
    train_stories = ['adollshouse', 'adventuresinsayingyes', 'alternateithicatom', 'avatar', 'buck', 'exorcism',
                'eyespy', 'fromboyhoodtofatherhood', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'inamoment',
                'itsabox', 'legacy', 'naked', 'odetostepfather', 'sloth',
                'souls', 'stagefright', 'swimmingwithastronauts', 'thatthingonmyarm', 'theclosetthatateeverything',
                'tildeath', 'undertheinfluence']

    test_stories = ["wheretheressmoke"]

    train_stories = []

    # Filter out the other stories for the tutorial
    for story in list(grids):
        if story not in (train_stories + test_stories):
            del grids[story]
            del trfiles[story]

    # Make datasequence for story
    wordseqs = make_word_ds(grids, trfiles)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We will be using a sliding context window with minimum size 256 words that increases until size 512 words.
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b") # Same tokenizer for all sizes
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b",
        torch_dtype=torch.float16,
        device_map={"": device}
    )

    # We will extract features from the 9th layer of the model
    layer_num = 9

    lookback1, lookback2 = 256, 512
    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, tokenizer, lookback1, lookback2
    )

    for word, compute_embedding in tqdm(compute_embeddings_flags.items()):
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

    feats = convert_to_feature_mats(
        wordseqs, word_to_tokens, token_sequence_registry
    )

if __name__ == "__main__":
    main()