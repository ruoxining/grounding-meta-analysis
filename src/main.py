"""The main module of the application."""
import argparse
import logging

from utils.experiment import Experiments

if __name__ == '__main__':
    # configs
    parser = argparse.ArgumentParser(description='Main module of the application.')
    parser.add_argument('--keyword', '-k', type=str, help='The keyword to search for.')
    parser.add_argument('--api_path', '-a', type=str, help='The path to the API key.')
    parser.add_argument('--experiment', '-e', type=str, help='The experiment to run.', choices=['all', 'keyword_model', 'topic_model', 'cooccuring_keywords', 'percent_numbers', 'complexity_scores', 'trend', 'semantic_change'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # main
    experiments = Experiments(keyword=args.keyword, api=args.api_path)

    # model word clusters
    if args.experiment == 'keyword_model' or args.experiment == 'all':
        experiments.model_word_clusters()

    # model topic clusters
    if args.experiment == 'topic_model' or args.experiment == 'all':
        experiments.model_topic_clusters()

    # model co-occurring keywords
    if args.experiment == 'cooccuring_keywords' or args.experiment == 'all':
        experiments.model_cooccurring_keywords()

    # model percent of numbers
    if args.experiment == 'percent_numbers' or args.experiment == 'all':
        experiments.model_percent_numbers()

    # model complexity scores
    if args.experiment == 'complexity_scores' or args.experiment == 'all':
        experiments.model_complexity_score()

    # model trend
    if args.experiment == 'trend' or args.experiment == 'all':
        experiments.model_trend()

    # model semantic change
    if args.experiment == 'semantic_change' or args.experiment == 'all':
        experiments.model_semantic_change()
