# kepler ps
# text preprocessing for sentiment scoring

import pandas as pd
# load in raw scraped data
df = pd.read_csv(r"\path\to\api_data")
# get list of all languages represented in the data

df['lang'].unique()

# filter out non-english tweets
df = df[df['lang'] == 'en']


# anonymizes each user and any mentioned links;
# tweet data structure requires checks when splitting on both spaces and new lines 
def preprocess(text):
    pass1 = []
    pass2 = []

    for t in text.split(" "):
        t = "[user]" if t.startswith("@") and len(t) > 1 else t
        print("Pass 1 user:", t)
        t = "[link]" if t.startswith("http") else t
        print("Pass 1 link:", t)
        pass1.append(t)
    pass1_joined = " ".join(pass1)
    print("Pass 1 appended:", pass1_joined)

    for t in pass1_joined.split("\n"):
        t = "[user]" if t.startswith("@") and len(t) > 1 else t
        print("Pass 2 user:", t)
        t = "[link]" if t.startswith("http") else t
        print("Pass 2 link:", t)
        pass2.append(t)
    print("Pass 2 appended:", pass2)

    return " ".join(pass2)


# apply the preprocessing function to the tweets, save in 'roberta_text' column
df["roberta_text"] = df["full_text"].apply(preprocess)

# check for mentions, website references 
df['roberta_text'][df.roberta_text.str.contains('@')]
df['roberta_text'][df.roberta_text.str.contains('http')]
df['roberta_text'].isna().sum()


# save data;
# keep record of index for batch processing later on
df.to_csv(r'\path\to\preprocessed_api_data', index=True)
