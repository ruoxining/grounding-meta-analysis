"""The main module of the application."""
import argparse
import logging

from utils.experiment import Experiments

if __name__ == '__main__':
    # configs
    parser = argparse.ArgumentParser(description='Main module of the application.')
    parser.add_argument('--keyword', '-k', type=str, help='The keyword to search for.')
    parser.add_argument('--api_path', '-a', type=str, help='The path to the API key.')
    parser.add_argument('--experiment', '-e', type=str, help='The experiment to run.', choices=['all'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # main
    experiments = Experiments(keyword=args.keyword)
    if args.experiment == 'keyword_model' or args.experiment == 'all':
        experiments.model_word_clusters()
