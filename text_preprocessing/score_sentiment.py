# kepler ps 
# sentiment scoring script

# packages
import pandas as pd
import numpy as np
import argparse
import torch  # For tensor operations with PyTorch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax


# takes text and model to calculate raw scores of each sentiment type (neg, neut, pos)
def model_scores(text, model):
    # tokenize text & apply model
    encoded_input = tokenizer(text, return_tensors="pt", max_length=514, padding='max_length',
                              truncation=True)  # larger max_lengths threw issues
    output = model(**encoded_input)
    print(output)
    # format scores
    raw_scores = output[0][0].detach().numpy()
    print(raw_scores)

    return raw_scores


# transforms raw sentiment scores to a single scalar value
def polarity(raw_scores):
    # apply softmax to raw scores
    scores_softmax = np.round(softmax(raw_scores), 2)
    # weight each probability, sum to a scalar
    polarity_weights = torch.tensor([-1, 0, 1])
    print("scores_softmax:", scores_softmax)
    probs = torch.tensor(scores_softmax)
    print("scores_tensor:", probs)
    polarity_score = polarity_weights * probs
    print("polarity by sign:", polarity)
    polarity_score = polarity_score.sum(dim=-1).numpy()
    print("type polarity sum", type(polarity))

    return polarity_score


# set up the parser 
parser = argparse.ArgumentParser()
parser.add_argument('-batch', type=int, help='Batch number')
args = parser.parse_args()

# get the preprocessed tweets
df = pd.read_csv(r'\path\to\preprocessed_api_data')

# subset tweets 
# you can manipulate the range of tweets to subset
if len(df) // ((args.batch + 1)*500000) == 0:
    subset = df.iloc[args.batch*500000:].copy()
else:
    subset = df.iloc[args.batch*500000:(args.batch + 1)*500000].copy()

# create an instance of the model (pretrained transformer layers & classification layer) 
model_path = u"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
roberta_model = AutoModelForSequenceClassification.from_pretrained(model_path)  # sequence classification head

# apply the sequence classification model
raw = subset["roberta_text"].apply(lambda t: model_scores(t, model=roberta_model)).tolist()
# transform, store raw scores
subset['polarity'] = [polarity(score) for score in raw]

# check output
print('assigned scores raw:', subset['polarity'])

# save to batch folder. keep index for combining batches later on
df.to_csv(rf"path\to\batch_{args.batch}.csv", index=True)
