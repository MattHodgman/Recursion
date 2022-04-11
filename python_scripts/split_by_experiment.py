# This script splits the deep learning embeddings and metadata by experiment and writes the files to each experiments directory.

import pandas as pd
import argparse
import constants


def parse_args():
    """
    Defines arguments.
    
    Returns:
        args: some sort of argparse object that the arguments can be extracted from.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--embeddings', help='CSV of DL embeddings.', type=str, required=True)
    parser.add_argument('-m', '--metadata', help='CSV of metadata.', type=str, required=True)
    parser.add_argument('-o', '--output', help='Directory to write output to. Default is current directory.', type=str, default='.', required=False)
    parser.add_argument('-n', '--outputFileName', help='Name of the ouput file.', type=str, default='embeddings_and_metadata.csv', required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Main."""

    # Parse arguments.
    args = parse_args()
    out_dir = args.output.rstrip('/')
    out_file_name = args.outputFileName

    # Load data.
    embeddings = pd.read_csv(args.embeddings)
    metadata = pd.read_csv(args.metadata)

    # Merge data.
    df = embeddings.merge(metadata, on=constants.SITE_ID)

    del embeddings
    del metadata

    # Split by experiment.
    experiments = list(df.experiment.unique())

    for exp in experiments:
        print(f'subsetting {exp} and writing to file {out_dir}/{exp}_{out_file_name}')
        exp_df = df[df[constants.EXPERIMENT] == exp]
        exp_df.to_csv(f'{out_dir}/{exp}_{out_file_name}', index=False)
