import pandas as pd
import pickle


def build_categorical_feature_dict(input_file, output_file):

    # Load the dataset
    data_df = pd.read_csv(input_file)

    # Transform categorical features to numerical labels such that the ML model can utilize those features properly
    # It is ok to transform columns that are already numerically-labelled, as long as they have the same encode treatment in prediction stage
    cat_feats = ['flight_no', 'Week', 'Departure','Arrival','Airline','std_hour']
    for cat_col in cat_feats:
        data_df[cat_col + "_label"] = data_df[cat_col].astype("category").cat.codes + 1

    # Get distinct categorical features and respective labels
    cat_and_label_feats = cat_feats + [cat + "_label" for cat in cat_feats]
    distinct_cat_label_df = data_df[cat_and_label_feats].drop_duplicates()

    # Encode as dict structure
    all_col_encode_dict = dict()
    for idx, row in distinct_cat_label_df.iterrows():
        for cat_col in cat_feats:
            original_val = row[cat_col]
            labelled_val = row[cat_col + "_label"]

            if cat_col not in all_col_encode_dict:
                all_col_encode_dict[cat_col] = dict()

            all_col_encode_dict[cat_col][original_val] = labelled_val

    # Save as file
    with open(output_file, 'wb') as f:
        pickle.dump(all_col_encode_dict, f, pickle.HIGHEST_PROTOCOL)

    return


def encode_categorical_feature(data_df, encode_dict_file_path):

    # To avoid side-effect
    to_return_df = data_df.copy()

    # Load encode dict file first
    all_col_encode_dict = None
    with open(encode_dict_file_path, 'rb') as f:
        all_col_encode_dict = pickle.load(f)

    # For the dataframe, replace cat. value per column with label
    cat_feats = ['flight_no', 'Week', 'Departure','Arrival','Airline','std_hour']

    def encode_feat(val, col_name):
        return all_col_encode_dict[col_name][val]

    for col in cat_feats:
        to_return_df[col] = to_return_df[col].apply(encode_feat, args=(col, ))

    return to_return_df
