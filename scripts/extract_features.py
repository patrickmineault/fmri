import joblib
from tqdm import tqdm
from ridge_utils.dsutils import make_word_ds
from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts, convert_to_feature_mats
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def fit_one_layer(model_name, layer_num):
    # These files are located in the story_data folder of the Box
    grids = joblib.load("/teamspace/uploads/grids_huge.jbl") # Load TextGrids containing story annotations
    trfiles = joblib.load("/teamspace/uploads/trfiles_huge.jbl") # Load TRFiles containing TR information

    # We'll build an encoding model using this set of stories for this tutorial.
    train_stories = ['adollshouse', 'adventuresinsayingyes', 'afatherscover', 'afearstrippedbare', 'againstthewind', 'alternateithicatom', 'avatar', 'backsideofthestorm', 'becomingindian', 'beneaththemushroomcloud', 'birthofanation', 'bluehope', 'breakingupintheageofgoogle', 'buck', 'canadageeseandddp', 'canplanetearthfeedtenbillionpeoplepart1', 'canplanetearthfeedtenbillionpeoplepart2', 'canplanetearthfeedtenbillionpeoplepart3', 'catfishingstrangerstofindmyself', 'cautioneating', 'christmas1940', 'cocoonoflove', 'comingofageondeathrow', 'escapingfromadirediagnosis', 'exorcism', 'eyespy', 'findingmyownrescuer', 'firetestforlove', 'food', 'forgettingfear', 'gangstersandcookies', 'goingthelibertyway', 'goldiethegoldfish', 'golfclubbing', 'googlingstrangersandkentuckybluegrass', 'gpsformylostidentity', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'ifthishaircouldtalk', 'igrewupinthewestborobaptistchurch', 'inamoment', 'indianapolis', 'itsabox', 'jugglingandjesus', 'kiksuya', 'lawsthatchokecreativity', 'learninghumanityfromdogs', 'leavingbaghdad', 'legacy', 'life', 'lifeanddeathontheoregontrail', 'lifereimagined', 'listo', 'marryamanwholoveshismother', 'mayorofthefreaks', 'metsmagic', 'mybackseatviewofagreatromance', 'myfathershands', 'naked', 'notontheusualtour', 'odetostepfather', 'penpal', 'quietfire', 'reachingoutbetweenthebars', 'seedpotatoesofleningrad', 'shoppinginchina', 'singlewomanseekingmanwich', 'sloth', 'souls', 'stagefright', 'stumblinginthedark', 'superheroesjustforeachother', 'sweetaspie', 'swimmingwithastronauts', 'tetris', 'thatthingonmyarm', 'theadvancedbeginner', 'theclosetthatateeverything', 'thecurse', 'thefreedomridersandme', 'theinterview', 'thepostmanalwayscalls', 'thesecrettomarriage', 'theshower', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'thetiniestbouquet', 'thetriangleshirtwaistconnection', 'threemonths', 'thumbsup', 'tildeath', 'treasureisland', 'undertheinfluence', 'vixenandtheussr', 'waitingtogo', 'whenmothersbullyback', 'whyimustspeakoutaboutclimatechange', 'wildwomenanddancingqueens']
    test_stories = ["wheretheressmoke", 'onapproachtopluto', 'fromboyhoodtofatherhood']
    #train_stories = ['stagefright']
    #test_stories = []

    # Filter out the other stories for the tutorial
    for story in list(grids):
        if story not in (train_stories + test_stories):
            del grids[story]
            del trfiles[story]

    # Make datasequence for story
    wordseqs = make_word_ds(grids, trfiles)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We will be using a sliding context window with minimum size 256 words that increases until size 512 words.
    tokenizer = AutoTokenizer.from_pretrained(model_name) # Same tokenizer for all sizes
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    model = model.to(device)

    if '2b' in model_name:
        lookback1, lookback2 = 256, 512
    elif '7b' in model_name:
        lookback1, lookback2 = 128, 256
    elif '27b' in model_name:
        lookback1, lookback2 = 32, 64
    else:
        raise NotImplementedError(f"Model {model_name} not supported for feature extraction.")

    word_to_tokens, compute_embeddings_flags, token_sequence_registry = generate_efficient_feat_dicts(
        wordseqs, tokenizer, lookback1, lookback2
    )

    for word, compute_embedding in tqdm(compute_embeddings_flags.items()):
        if compute_embedding:
            inputs = {"input_ids": torch.tensor([word_to_tokens[word]], device=device).int()}
            inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, device=device)
            out = list(model(**inputs, output_hidden_states=True)[2])
            out = out[layer_num][0].detach().cpu().numpy()
            this_key = tuple(inputs["input_ids"][0].cpu().numpy())
            for ei, _ in enumerate(this_key):
                prefix = this_key[: ei + 1]
                if prefix in token_sequence_registry:
                    token_sequence_registry[prefix] = out[ei, :]

    feats = convert_to_feature_mats(
        wordseqs, word_to_tokens, token_sequence_registry
    )

    with open(f"../data/word_features_{model_name.replace('/', '_')}_layer{layer_num:02}.pkl", "wb") as f:
        joblib.dump(feats, f)

if __name__ == "__main__":
    for model_name, layers in [
        #("google/gemma-2-2b", range(0, 26)),
        #("google/gemma-2-9b", range(0, 32)),
        #("google/gemma-2-9b-it", [9, 20, 31]),
        ("google/gemma-2-27b", [10, 22, 34])
    ]:
        for layer_num in layers:
            long_name = f"{model_name}_layer{layer_num:02}"
            print(f"Fitting model {long_name}")
            fit_one_layer(model_name, layer_num)