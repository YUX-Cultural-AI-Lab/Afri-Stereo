import pandas as pd
import re  # regex for pattern matching
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

import json # save dictionary

import argparse # to get the file name as an argument from the script

FILE_PATH = ''

def preprocess_data(df):
    # this function takes the raw survey data and brings it into a form that is handleable
    # input: df -> dataframe consisting raw survey data
    # output: df -> appropriately modified dataframe

    # drop irrelevant columns
    df = df.drop(["Nom d'étude", "UID de l'interview", "User ID", "Enquêteur", "Telephone", 'Email', "Durée de l'interview",
                "Date de création", "Date de modification", "Repondant", "Adresse IP", "Pays", "Ville", "Region", "Statut"], axis=1)
    
    # rename columns to relevant shorthand
    df.rename({
        'S01_Q01.What is your gender identity?': 'gender_identity',
        'S01_Q02. What is your age range?': 'age_range',
        'S01_Q03.Are you currently:': 'employment_status',
        'S01_Q04.Which sector do you work in?': 'sector',
        'S01_Q05.What is your religion or belief system?': 'religion',
        'S01_Q06.What country do you live in?': 'country',
        'S01_Q07.What is your nationality?': 'nationality',
        'S01_Q08. What ethnic group(s) in Nigeria do you identify with? (Select all that apply)': 'ethnic_nigeria',
        'S01_Q09. What ethnic group(s) in Kenya do you identify with? (Select all that apply)': 'ethnic_kenya'
    }, axis=1, inplace=True)

    df.rename({
    'S03_Q03.What are some of the common stereotypes associated with women? For example, "Women are nurturing". Please provide as many examples as you’d like — just separate each one with a comma.': 'stereotypes_women',

    'S03_Q04.What are some of the common stereotypes associated with men? For example,  "Men are strong." Please provide as many examples as you’d like — just separate each one with a comma.': 'stereotypes_men',

    'S03_Q05.What are some of the common stereotypes associated with people\'s ethnicity or regions? For example, "People from [XYZ ethnic group] are aggressive." "People from [XYZ ethnic group] are uneducated.", "People from the north are...", "People from the east are..."  . Please provide as many examples as you’d like — just separate each one with a comma.': 'stereotypes_ethnicity_region',

    'S03_Q06.What are some common stereotypes associated with people\'s religion? For example, "Muslims are ....", "Christians are...", "Traditional worshippers are ...".  Please provide as many examples as you’d like — just separate each one with a comma.': 'stereotypes_religion',

    'S03_Q07.What are some common stereotypes associated with people\'s age? For example, "Old people are wise", "Young people are careless". Please provide as many examples as you’d like — just separate each one with a comma.': 'stereotypes_age',

    'S03_Q08.What are some common stereotypes associated with people\'s professions? For example, "Doctors are smart", "Traders are persuasive".  Please provide as many examples as you’d like — just separate each one with a comma.': 'stereotypes_profession',

    'S03_Q09.Do you know of any other stereotypes commonly associated with different groups of people?  These could include stereotypes related to ethnicity, gender, profession, or any other group you can think of. Please provide as many examples as you’d like — just separate each one with a comma.\n': 'stereotypes_misc'
    }, axis=1, inplace=True)

    # merge redundant columns
    df['ethnic_group'] = df['S01_Q11.What ethnic group do you identify with?'].fillna('') + ' ' + df['S01_Q11.What ethnic group do you identify with?.1'].fillna('')
    df['ethnic_group'] = df['ethnic_group'].str.strip()  # Clean up extra spaces
    df.drop(['S01_Q11.What ethnic group do you identify with?',
            'S01_Q11.What ethnic group do you identify with?.1'], axis=1, inplace=True)

    # drop columns that we don't need
    df.drop([
        'S03_Q01.Before we continue, please note that the next few questions will ask about common stereotypes or generalisations you may have heard in your community. You don’t have to personally believe these views — we’re simply trying to understand the kinds of common perspectives people may have about others based on things like age, gender, religion, ethnic group, profession, or other social identities. Some of these may be offensive or uncomfortable, but please remember your responses are anonymous, and nothing you say will be used against you.',

        'S03_Q02.The stereotypes should be structured using an IDENTITY TERM, such as "Women," "Christians," or "Doctors," paired with an ATTRIBUTE TERM, which is an adjective that describes a characteristic or trait attributed to that group. For example: "Women are nurturing," where "Women" is the identity and "nurturing" is the attribute; "Men are strong," with "Men" as the identity and "strong" as the attribute; "Old people are wise," where "Old people" is the identity and "wise" is the attribute; "Doctors are smart," with "Doctors" as the identity and "smart" as the attribute; or "People from [XYZ ethnic group] are aggressive," where "[XYZ ethnic group]" is the identity and "aggressive" is the attribute.'
    ], axis=1, inplace=True)

    return df

def detect_language(text):
    # simple function to detect the language of a text using the detect function from the LangDetect module
    # input -> text: any string. 
    # output -> returns a language code such as "en" for english, "fr for french", etc and "unkwown" if it cannot identify it
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text):
    # simple function that uses google translator to translate a text from non-english to english
    # input -> text: any string
    # output -> returns that language translated to english, or the string as is if the translation doesn't go through
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text  # fallback if translation fails

def form_initial_stereo_dataset(df):
    # given the cleaned up survey responses, we now bring the dataset into a form that has one stereotype per row. 
    # information per row includes the stereotype sentence submitted by the respondent and the demographic details of the respondent
    # this also identifies seemingly non-english entries and translates it
    # input: df -> cleaned up survey responses
    # output: writes stereotype_df into ../data/processed/translated_stereotypes.csv

    # using the category of stereotype, convert the dataset such that each row has a stereotype
    stereotype_columns = [
    'stereotypes_women',
    'stereotypes_men',
    'stereotypes_ethnicity_region',
    'stereotypes_religion',
    'stereotypes_age',
    'stereotypes_profession',
    'stereotypes_misc'
    ]

    # Columns to keep as metadata (e.g. gender, country, etc.)
    metadata_columns = [col for col in df.columns if col not in stereotype_columns]

    # we break the data such that each row has 1 stereotype. for every individual, we separate all their stereotypes and put them into separate rows.
    # we write this into the stereotype_df dataframe

    rows = []

    for idx, row in df.iterrows():
        for col in stereotype_columns:
            cell = row.get(col)
            col_name = col[12:]
            if pd.notna(cell):  # only if there's content in the cell
                stereotypes = [s.strip() for s in re.split(r'[,\n]+', cell) if s.strip()]
                for stereotype in stereotypes:
                    entry = {
                        'stereotype_sentence': stereotype,
                        'stereotype_type': col_name,
                        'original_idx': idx
                    }
                    # Include metadata from original row
                    for meta_col in metadata_columns:
                        entry[meta_col] = row[meta_col]
                    rows.append(entry)

    stereotype_df = pd.DataFrame(rows)

    stereotype_df['lang'] = stereotype_df['stereotype_sentence'].apply(detect_language) # identify non-english rows of the dataframe

    # Count and log number of 'unknown' language rows
    unknown_lang_count = (stereotype_df['lang'] == 'unknown').sum()
    print(f"Dropping {unknown_lang_count} rows with 'unknown' language.")

    # Drop entries with unknown language
    stereotype_df = stereotype_df[stereotype_df['lang'] != 'unknown'].copy()

    # Count non-English rows before translating
    non_english_count = (stereotype_df['lang'] != 'en').sum()
    print(f"Translating {non_english_count} non-English rows to English.")

    # Translate only non-English rows
    stereotype_df['translated_stereotype'] = stereotype_df.apply(
        lambda row: translate_to_english(row['stereotype_sentence']) if row['lang'] != 'en' else row['stereotype_sentence'],
        axis=1
    )

    # we merge these three columns to form one ethnic_identity column (only 1 of them is filled always)
    stereotype_df['ethnic_identity'] = (
        stereotype_df[['ethnic_nigeria', 'ethnic_kenya', 'ethnic_group']]
        .bfill(axis=1)
        .iloc[:, 0]
    )
    stereotype_df.drop(columns=['ethnic_nigeria', 'ethnic_kenya', 'ethnic_group'], inplace=True)

    stereotype_df.to_csv('../data/processed/translated_stereotypes.csv', index=False)
    print(f"Successfully stored the initial stereotype dataframe at data/processed/translated_stereotypes.csv")

def expand_rows_by_sentences(df):
    # this function expands the entries that have multiple stereotypes per row (separated by a ./!/?) into separate entries
    # input: df -> the output after cleaning up that has been read from translated_stereotypes.csv
    # output: output dataframe based on the expanded rows from translated_stereotype
    expanded_rows = []
    for _, row in df.iterrows():
        # safely convert to string and split on '.', '!', or '?'
        sentences = [s.strip() for s in re.split(r'[.!?]', str(row['translated_stereotype'])) if s.strip()]
        for sentence in sentences:
            new_row = row.copy()
            new_row['translated_stereotype'] = sentence
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

def extract_regex(sentence, stereotype_type=None):
    # this is the function that extracts the identity term and the attribute term from the stereotype using regular expressions
    # input: sentence -> the stereotype sentence sent in by the respondent
    # stereotype_type ->  specifies the stereotype type for the regex
    # output: returns (identity, attribute) from the stereotype based on these rules

    # we also extract the known identity terms to aid with the identity extraction. 
    # load the identity terms
    with open("../data/raw/identity_categories.json", "r", encoding="utf-8") as f:
        identity_categories = json.load(f)

    # flatten the set of all identity terms
    known_identities = sorted(set(term.lower() for terms in identity_categories.values() for term in terms), key=len, reverse=True)

    sentence = sentence.strip()

    # Pattern 1: "People from [the] XYZ ..."
    match = re.match(r"^(People from(?: the)?\s+[A-Za-z\s\-']+?)\s+(.*)", sentence, re.IGNORECASE)
    if match:
        identity = match.group(1).strip().lower()
        attribute = match.group(2).strip().rstrip('.').lower()
        return identity, attribute

    # Pattern 1b: "[XYZ] people ..."
    if "people" in sentence.lower():
        match = re.match(r"^(.*?\bpeople\b)\s+(.*)", sentence, re.IGNORECASE)
        if match:
            identity = match.group(1).strip().lower()
            attribute = match.group(2).strip().rstrip('.').lower()
            return identity, attribute

    # Pattern 2: One word and known stereotype type
    words = sentence.split()
    if len(words) == 1 and stereotype_type in ['men', 'women']:
        return stereotype_type, words[0].lower()

    # Pattern 2b: "They are ..." or "They ..." (for men/women)
    if stereotype_type in ['men', 'women']:
        match = re.match(r"^they(?:\s+are)?\s+(.*)", sentence, re.IGNORECASE)
        if match:
            return stereotype_type, match.group(1).strip().rstrip('.').lower()

    # Pattern 3: Common structure like "X are Y"
    match = re.match(
        r"^(.*?)\s+(are|is|have|has|should be|can be|tend to be|must be|will be|were|was)\s+(.*)",
        sentence,
        re.IGNORECASE
    )
    if match:
        identity = match.group(1).strip().lower()
        attribute = match.group(3).strip().rstrip('.').lower()
        return identity, attribute

    # Pattern 4: Fallback using known identity list (whole words only)
    sentence_lower = sentence.lower()
    for identity in known_identities:
        if re.search(rf'\b{re.escape(identity)}\b', sentence_lower):
            idx = sentence_lower.find(identity)
            after = sentence_lower[idx + len(identity):].strip()
            after = re.sub(r'^(are|is|have|has|should be|can be|tend to be|must be|were|was)\s+', '', after)
            return identity, after.strip().rstrip('.')

    # Pattern 5: First word = identity, rest = attribute
    if len(words) > 1:
        return words[0].lower(), ' '.join(words[1:]).strip().rstrip('.').lower()

    return None, None

def extract_identities(entry):
    # the identity column consists of multiple identity terms separated by a *, we extract the separate ones and return them as a list
    # for example, "Kikuyu * Luo *" becomes ['kikuyu', 'luo']
    # input: entry -> the input string (in the example, "Kikuyu * Luo *")
    # output: list of terms (in this example, ['kikuyu', 'luo'])
    if pd.isna(entry):
        return []
    # Split by '*', ',', or ' and ' (with optional spaces)
    parts = re.split(r'\s*(?:\*|,| and )\s*', entry)
    return [p.strip().lower() for p in parts if p.strip()]



def custom_cleanup(stereotype_df):
    # this function contains all the custom clean up rules that I used on the dataframe. 
    # These contain a bunch of hard-coded rules, that are not generalizable, but were needed to process the data. 
    # Upon examination of the outputs, further custom rules can be added to this function based on the data. 
    # input: stereotype_df -> stereotype dataframe after the identity, attribute terms have been extracted
    # output: stereotype_df -> outputs the dataframe after performing custom cleaning up operations as we have defined

    # Drop rows where 'attribute_term' contains the exact string 'people from [XYZ ethnic group]'
    stereotype_df = stereotype_df[~stereotype_df['translated_stereotype'].str.contains('people from [XYZ ethnic group]', case = False, regex=False)]

    # upon manual inspection, this entry seems to be GPT generated and is corrupt
    stereotype_df = stereotype_df[stereotype_df['original_idx'] != 58]

    # we clean up the identity terms and normalize them    
    identity_replacements = {
    'caucasienne': 'caucasian',
    'les bretons !': 'caucasian',
    'black african': 'other_black',
    'black': 'other_black',
    '-': 'other',
    'none': 'other',
    'multiple': 'other'}

    # Function to apply replacements to each list
    def normalize_identity_list(identity_list):
        return [identity_replacements.get(term, term) for term in identity_list]

    # we apply this function to the dataframe
    stereotype_df['ethnic_identity'] = stereotype_df['ethnic_identity'].apply(normalize_identity_list)

    # we normalize the gender identity column by removing the '*' and stripping it of extra whitespaces
    stereotype_df['gender_identity'] = stereotype_df['gender_identity'].str.replace(r'\s*\*$', '', regex=True).str.strip()

    # we do this for the religion, nationality, and country columns as well
    stereotype_df['religion'] = stereotype_df['religion'].str.replace(r'\s*\*$', '', regex=True).str.strip()
    stereotype_df['nationality'] = stereotype_df['nationality'].str.replace(r'\s*\*$', '', regex=True).str.strip()
    stereotype_df['country'] = stereotype_df['country'].str.replace(r'\s*\*$', '', regex=True).str.strip()

    # we standardize these religious entries 
    stereotype_df['religion'] = stereotype_df['religion'].replace({
        'agnostic': 'Agnostic',
        'Agnostique': 'Agnostic',
        'Spiritual (not specific)': 'Other',
        'African Traditional religion (ATR)': 'Other'
    })

    # standardize nationality column
    stereotype_df['nationality'] = stereotype_df['nationality'].apply(
        lambda x: x if x in ['Nigerian', 'Kenyan', 'Senegalese'] else 'Other Nationality')
    
    # standardize country column
    stereotype_df['country'] = stereotype_df['country'].apply(
        lambda x: x if x in ['Nigeria', 'Kenya', 'Senegal'] else 'Other Country'
    )

    # drop entries where the identity was not extracted (is NaN)
    stereotype_df = stereotype_df[~stereotype_df['identity_term'].isna()].copy()

    return stereotype_df

def normalize_identity(identity):
    # we do some normalizations on the identity term extracted by the regex

    if pd.isna(identity):
        return identity

    identity = identity.lower().strip()

    if 'yoruba' in identity:
        return 'yoruba'
    elif 'hausa' in identity:
        return 'hausa'
    elif 'igbo' in identity:
        return 'igbo'
    elif 'the elderly' in identity:
        return 'old people'
    elif identity == 'christian':
        return 'christians'
    elif identity == 'moslems':
        return 'muslims'
    elif identity == 'the senegalese':
        return 'senegalese'
    elif identity == 'woman':
        return 'women'
    else:
        return identity

def save_outputs(stereotype_df):
    # we finally use this function to save the rare identity terms and the rest of the dataframe separately.
    # input: stereotype_df-> the dataframe consisting of the stereotypes after all the normalizations and the terms extracted
    # output: we save 2 files. 1) is all the entries with rare stereotypes (<=2 occurrences). 2) is all the entries with >2 occurrences. 

    # identify rarely occurring identities (≤ 2)
    identity_counts = stereotype_df['identity_term'].value_counts()
    rare_identities = identity_counts[identity_counts <= 2].index

    # save rows with rare identities for manual checking
    rare_identity_df = stereotype_df[stereotype_df['identity_term'].isin(rare_identities)]
    rare_identity_df.to_csv("../data/processed/stereotypes_with_rare_identities.csv", index=False)

    # keep only the rows with frequent identities
    stereotype_df = stereotype_df[stereotype_df['identity_term'].isin(identity_counts[identity_counts > 2].index)]

    # remove the spurious identity terms that might have been extracted
    spurious_identities = ['the', 'not', 'no', 'they', 'always']
    stereotype_df = stereotype_df[~stereotype_df['identity_term'].isin(spurious_identities)]

    # save the final extracted stereotypes (that are not rare)
    stereotype_df.to_csv("../data/processed/final_extracted_stereotypes.csv")

    print("The outputs have been stored into ../data/processed/stereotypes_with_rare_identities.csv and ../data/processed/final_extracted_stereotypes.csv.")

def main():
    parser = argparse.ArgumentParser(description="Script to process a file.")
    parser.add_argument("file_path", nargs="?", type=str, 
                        default="../data/raw/afristereo_survey_responses.csv", help="Path to the input file")
    parser.add_argument("--no_recompute_initial", action="store_false", dest="recompute_initial",
                    help="If set, skips recomputing the initial data")

    args = parser.parse_args()

    DetectorFactory.seed = 0 # ensures consistent results

    if args.recompute_initial:
        df = pd.read_csv(args.file_path) # read the dataframe containing the raw survey responses
        df = preprocess_data(df) # bring the dataframe into an easily understandable form
        form_initial_stereo_dataset(df) # form the stereo dataset and save it into the required path
    
    stereotype_df = pd.read_csv('../data/processed/translated_stereotypes.csv')
    stereotype_df = expand_rows_by_sentences(stereotype_df)

    # Apply regex
    stereotype_df[['identity_term', 'attribute_term']] = stereotype_df.apply(
        lambda row: pd.Series(extract_regex(row['translated_stereotype'], row['stereotype_type'])),
        axis=1)
    
    # we format the ethnic identity term in the required manner
    stereotype_df['ethnic_identity'] = stereotype_df['ethnic_identity'].apply(extract_identities)

    # we perform the custom cleanup on the stereotype_df
    stereotype_df = custom_cleanup(stereotype_df)

    # Apply the identity normalization
    stereotype_df['identity_term'] = stereotype_df['identity_term'].apply(normalize_identity)

    save_outputs(stereotype_df) # finally, save the outputs

if __name__ == "__main__":
    main()