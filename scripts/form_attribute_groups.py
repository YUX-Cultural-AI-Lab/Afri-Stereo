import pandas as pd
import re  # regex for pattern matching

from sentence_transformers import SentenceTransformer # embedding model
from sklearn.metrics.pairwise import cosine_similarity # cosine similarity calculator
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer # sia to calculate polarity of a statement

import json # save dictionary
import argparse # to get the file name as an argument from the script

def custom_preprocess(extracted_stereotypes):
    # this function takes the extracted stereotypes files after the human annotation, and performs a clean-up and replaces a few terms before returning it. 
    # Note: The replacements are custom, and might need to be modified appropriately. 
    # input: extracted_stereotypes -> dataframe formed by reading the outputs of the annotators
    # output: extracted_stereotypes -> after it has been modified

    # Treat blank or whitespace-only strings as NaN in the modified columns
    extracted_stereotypes["identity_term_modified"] = extracted_stereotypes["identity_term_modified"].replace(r"^\s*$", np.nan, regex=True)
    extracted_stereotypes["attribute_term_modified"] = extracted_stereotypes["attribute_term_modified"].replace(r"^\s*$", np.nan, regex=True)

    # Now safely use combine_first
    extracted_stereotypes["identity_term"] = extracted_stereotypes["identity_term_modified"].combine_first(extracted_stereotypes["identity_term"])
    extracted_stereotypes["attribute_term"] = extracted_stereotypes["attribute_term_modified"].combine_first(extracted_stereotypes["attribute_term"])

    # Forward fill metadata only where stereotype_sentence is NaN
    metadata_cols = [
        "stereotype_sentence", "stereotype_type", "original_idx", "gender_identity", "age_range",
        "employment_status", "sector", "religion", "country", "nationality", "lang", "translated_stereotype", "ethnic_identity"
    ]
    for col in metadata_cols:
        mask = extracted_stereotypes["stereotype_sentence"].isna()
        extracted_stereotypes.loc[mask, col] = extracted_stereotypes[col].ffill()[mask]

    # Drop the modified columns
    extracted_stereotypes = extracted_stereotypes.drop(columns=["identity_term_modified", "attribute_term_modified"])
    extracted_stereotypes = extracted_stereotypes.dropna(subset=["identity_term", "attribute_term"])

    extracted_stereotypes["identity_term"] = (
    extracted_stereotypes["identity_term"]
    .str.strip()      # remove leading/trailing spaces
    .str.lower()      # convert to lowercase
    )

    # Define the replacements. NOTE: These are specific replacements based on the data, and are NOT generalizable. 
    # This will need to be modified accordingly. Ideally, at least some of these issues should be resolved in the annotation stage, this was following
    # another sanity check
    replacements = {
        "people from the north": "people from northern nigeria",
        "people from the south": "people from southern nigeria",
        "people from the east": "people from eastern nigeria",
        "people from the west": "people from western nigeria",
        "people with tatoos": "people with tattoos",
        "peulhs people": "peulh people",
        "northerners": "people from northern nigeria",
        "city peole": "city people",
        "the peulhs": "peulh people",
        "yoruba": "yoruba people",
        "igbo": "igbo people",
        "males": "men",
        "females": "women",
        "human doctors": "doctors"
        }

    # Apply the replacements in-place
    extracted_stereotypes["identity_term"] = extracted_stereotypes["identity_term"].replace(replacements)

    return extracted_stereotypes

# group attribute terms together based on their similarity in embedding space. the idea is that similar/synonymous stereotypes will get grouped together

def group_attributes_by_identity(identity_df, model, col_name = 'attribute_term', threshold=0.55):
    # this function takes the stereotype dataframe and the embedding model, and uses a threshold to group elements of a column together based on their 
    # cosine similarities
    # input: identity_df -> the stereotype dataframe we are dealing with
    # model -> the embedding model used for the grouping
    # col_name -> column we are interested in, i.e. containing the attribute terms/phrases
    # threshold -> value above which the cosine similarity score must go for a pair of words to be classified as synonyms. THIS IS TUNABLE.
    # output: sim_matrix -> cosine similarity matrix between any two of the attribute terms/phrases
    # grouped -> returns a list of grouped terms based on the threshold

    attrs = identity_df[col_name].unique()
    attrs = [a for a in attrs if a]  # remove blanks

    if len(attrs) == 0:
        return []

    embeddings = model.encode(attrs) # obtain all embeddings of this column
    sim_matrix = cosine_similarity(embeddings) # form the similarity matrix which is a N x N matrix with cosine similarities.

    # simple greedy grouping based on similarity threshold
    grouped = []
    visited = set()

    # iterate through the attributes
    for i, attr in enumerate(attrs):

        if i in visited:
          # if the attribute is already in a group, we ignore
          continue

        # if not, we form a single member group with just this attribute
        group = [attr]
        visited.add(i)

        # for each attribute, iterate through the remaining unvisited attributes, and add them to the existing group if their similarity meets the threshold requirements
        for j in range(i + 1, len(attrs)):
            if j not in visited and sim_matrix[i, j] >= threshold:
                group.append(attrs[j])
                visited.add(j)

        # make a collection of all these groups
        grouped.append(group)

    # return the similarity matrix and the list of groups
    return sim_matrix, grouped

def get_polarity(phrase, sia_model, threshold = 0.1):
    # this function takes a phrase and a sentiment intensity analyzer (SIA) model, and a threshold, and uses it to classify its sentiment
    # input: phrase -> a particular term/phrase
    # sia_model -> the sentiment intensity analyzer model
    # threshold -> dictates how much you are willing to classify something as 'neutral'. This is TUNABLE. 
    # A lower value of threshold means that your classification as positive/negative sentiment is more assertive
    score = sia_model.polarity_scores(phrase)['compound']
    if score >= threshold:
        return 'positive'
    elif score <= -threshold:
        return 'negative'
    else:
        return 'neutral'

def split_group_by_polarity(group, sia_model):
    # split each group into subgroups based on polarity
    # for example: ['happy', 'very happy', 'sad'] gets split into ['sad'] and ['happy', 'very happy']
    polarity_buckets = {'positive': [], 'negative': [], 'neutral': []}
    for phrase in group:
        polarity = get_polarity(phrase, sia_model)
        polarity_buckets[polarity].append(phrase)
    # return non-empty buckets
    return [phrases for phrases in polarity_buckets.values() if phrases]


def main():
    parser = argparse.ArgumentParser(description="Script to process a file.")
    parser.add_argument("file_paths", nargs="+", type=str,
                    help="One or more input file paths (space-separated)")

    args = parser.parse_args()

    # read all the stereotypes and concatenate them to form the extracted_stereotypes dataframe
    extracted_stereotypes = pd.concat( [pd.read_csv(file_path) for file_path in args.file_paths], ignore_index=True)

    # cleanup the stereotype file
    extracted_stereotypes = custom_preprocess(extracted_stereotypes)

    # load a good, generalizable, lightweight embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2") # ISSUE: this model doesnt detect positives/negatives and groups them together

    _, grouped = group_attributes_by_identity(extracted_stereotypes, model) # return the groupings 

    # Initialize polarity analyzer (to separate the above groups based on polarity)
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    polarity_split_groups = []
    for group in grouped:
        subgroups = split_group_by_polarity(group, sia)
        polarity_split_groups.extend(subgroups)

    # define the groups we had as a dictionary that maps an attribute to what group it belongs to, so that we can speedup lookup later
    attribute_to_group = {}
    for group in polarity_split_groups:
        group_tuple = tuple(group)
        for attr in group:
            attribute_to_group[attr] = group_tuple

    # save this dictionary as a json file
    with open("./data/processed/attribute_to_group_initial.json", "w") as f:
        json.dump({k: list(v) for k, v in attribute_to_group.items()}, f)
    
    print('The attribute group dictionary has been saved to data/processed/attribute_to_group_initial.json!')


if __name__ == "__main__":
    main()