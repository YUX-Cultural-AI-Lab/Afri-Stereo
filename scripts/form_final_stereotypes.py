import argparse
import pandas as pd
import json

def form_stereotype_output(stereotype_df, attribute_to_group):
    unique_religions = ['Christianity', 'Atheism', 'Agnostic', 'Islam', 'Other']
    ethnic_list = ['caucasian', 'delta igbo', 'ebira', 'fon', 'fulani', 'gamai', 'goemai', 'hausa', 'ibibio', 'idoma', 'igala', 'igbira', 'igbo', 'ijaw', 'ishan', 'jukun', 'kikuyu', 'luhya', 'luo', 'mossi', 'mwaghavul', 'nupe', 'other', 'other_black', 'peulh', 'serrere', 'tiv', 'ukwuani', 'urhobo', 'wolof', 'yoruba']
    
    output_columns = (
    ['Total', 'Male', 'Female'] +
    unique_religions +
     ethnic_list +
    ['Nigerian', 'Kenyan', 'Senegalese', 'Other Nationality'] +
    ['Nigeria', 'Kenya', 'Senegal', 'Other Country']
    )

    # Initialize empty list for rows
    rows = []

    # dictionary for fast lookup of (identity_term, attribute_group) rows
    lookup = {}

    # iterate through each row of stereotype_df
    for idx, row in stereotype_df.iterrows():
        identity = row['identity_term']
        attr = row['attribute_term']
        gender = row['gender_identity']
        religion = row['religion']
        country = row['country']
        nationality = row['nationality']
        ethnic_list = row['ethnic_identity'] if isinstance(row['ethnic_identity'], list) else []

        # find the attribute group tuple for this attribute_term
        # if not found, use attribute_term itself as a single-element tuple
        attribute_group = attribute_to_group.get(attr, (attr,))

        key = (identity, attribute_group) # form the key for the lookup dictionary

        if key not in lookup:
            # if this key does not already exist (i.e. it is a newly encountered stereotype)
            # then, we initialize counts for all output columns to zero.
            data = {col: 0 for col in output_columns}
            data['Total'] = 0
            data['identity_term'] = identity
            data['attribute_group'] = list(attribute_group)
            lookup[key] = data
            rows.append(data)

        # then, we update the total counts, and also counts across different categories based on the identity of the respondent

        # update total count
        data = lookup[key]
        data['Total'] += 1

        # update gender count
        if gender in ['Male', 'Female']:
            data[gender] += 1

        # update religion count if in list
        if religion in unique_religions:
            data[religion] += 1

        # update country count if in list
        if country in ['Nigeria', 'Kenya', 'Senegal', 'Other Country']:
            data[country] += 1

        # update nationality count if in list
        if nationality in ['Nigerian', 'Kenyan', 'Senegalese', 'Other Nationality']:
            data[nationality] += 1

        # update ethnic_identity counts (iterating through all possible identities a person might have, which is stored as a list)
        for ethnic in ethnic_list:
            data[ethnic] += 1

    # Convert list of rows to DataFrame
    output_df = pd.DataFrame(rows)

    cols_order = ['identity_term', 'attribute_group', 'Total'] + ['Male', 'Female'] + [col for col in output_columns if col not in ['Total', 'Male', 'Female']]
    cols_order = [c for c in cols_order if c in output_df.columns]  # make sure columns exist
    output_df = output_df[cols_order]

    output_df = output_df.sort_values(by='Total', ascending=False).reset_index(drop=True)

    return output_df


def main():
    parser = argparse.ArgumentParser(description="Script to process a file.")
    parser.add_argument("grouping_path", nargs="?", type=str, 
                        default="./data/processed/attribute_to_group_modified.json", help="Path to the input file")
    args = parser.parse_args()
    
    stereotype_df = pd.read_csv("./data/processed/final_cleaned_stereotypes.csv")

    with open(args.grouping_path, "r", encoding="utf-8") as f:
        raw_dict = json.load(f)
        attribute_to_group = {k: tuple(v) for k, v in raw_dict.items()}
    
    output_df = form_stereotype_output(stereotype_df, attribute_to_group)

    output_df.to_csv("./data/processed/stereotype_summary.csv")

    print('Final output successfully written into data/processed/stereotype_summary.csv!')

if __name__ == "__main__":
    main()
    






    
