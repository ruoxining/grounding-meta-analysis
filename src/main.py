"""The main module of the application."""
import argparse
import logging

from utils.experiments import Experiments

if __name__ == '__main__':
    # configs
    parser = argparse.ArgumentParser(description='Main module of the application.')
    parser.add_argument('--keyword', '-k', type=str, help='The keyword to search for.')
    parser.add_argument('--api_path', '-a', type=str, help='The path to the API key.')
    parser.add_argument('--experiment', '-e', type=str, help='The experiment to run.', choices=['keyword_model', 'topic_model', 'trend_graph', 'complexity_score', 'dependency', 'all'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # main
    experiments = Experiments(keyword=args.keyword)
    if args.experiment == 'keyword_model' or args.experiment == 'all':
        experiments.model_word_clusters()
    elif args.experiment == 'topic_model' or args.experiment == 'all':
        experiments.model_topic_modeling()
    elif args.experiment == 'trend_graph' or args.experiment == 'all':
        experiments.model_trend_graph()
    elif args.experiment == 'complexity_score' or args.experiment == 'all':
        experiments.model_complexity_score()
    elif args.experiment == 'dependency' or args.experiment == 'all':
        experiments.model_dependency()
    else:
        raise ValueError("Invalid experiment.")
