# Power transform each plate separately to remove batch effects.

# Load packages.
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import argparse
import constants


def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='Power normalize deep learning features and return normalized features.')
    parser.add_argument('-i', '--input', help='CSV of DL embeddings and metadata for a specific experiment.', type=str, required=True)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)
    parser.add_argument('-n', '--outputFileName', help='Name of the ouput file.', type=str, default='normalized_embeddings_and_metadata.csv', required=False)
    args = parser.parse_args()
    return args


def power_transform(df, feature_cols):
    """Power transform by plate"""

    # break df into a separate array for each plate
    plates = df[constants.PLATE].unique()
    plate_values = {}
    plate_metadata = {}

    for plate in plates:
        values = df[df[constants.PLATE] == plate][feature_cols].values
        plate_values[plate] = values
        plate_metadata[plate] = df.loc[:, ~df.columns.isin(feature_cols)]

    plate_transformed_dfs = []

    for plate, values in plate_values.items():
        pt = PowerTransformer(method='yeo-johnson')  # Yeo-Johnson can handle positive and negative numbers
        transformed_values = pt.fit_transform(values)
        plate_transformed_df = pd.DataFrame(transformed_values, columns=feature_cols)
        plate_transformed_df = plate_transformed_df.join(plate_metadata[plate].reset_index())
        plate_transformed_dfs.append(plate_transformed_df)

    # combine transformed plate dfs into one
    plate_transformed_df = plate_transformed_dfs[0]
    for i in range(1, len(plate_transformed_dfs)):
        plate_transformed_df = plate_transformed_df.append(plate_transformed_dfs[i], ignore_index=True)

    return plate_transformed_df


if __name__ == '__main__':
    """Main."""

    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    out_file_name = args.outputFileName

    # Load data and preprocess data.
    df = pd.read_csv(args.input)
    df[constants.DISEASE_CONDITION] = df[constants.DISEASE_CONDITION].fillna(constants.NULL)
    feature_cols = [col for col in df.columns if constants.FEATURE_PREFIX in col]
    cols_to_keep = feature_cols + [constants.PLATE] + [constants.DISEASE_CONDITION]

    # Normalize
    plate_transformed_df = power_transform(df, feature_cols)
    plate_transformed_df.to_csv(f'{out_dir}/{out_file_name}')
