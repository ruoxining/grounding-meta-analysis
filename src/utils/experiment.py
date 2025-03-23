"""Class to conduct experiments."""
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from .dataset import PaperDataset
from .querier import Querier


class Experiments:
    """Class to conduct experiments."""
    def __init__(self, keyword: str, api: str) -> None:
        """Initialize the Experiments class."""
        self._keyword = keyword
        self._paper_dataset = PaperDataset(keyword=keyword)
        self._querier = Querier(model_name='gpt-4', api_key=api)
        self._encoder = SentenceTransformer('all-mpnet-base-v2')

    def model_word_clusters(self) -> None:
        """Model the keywords extracted from the paper through clustering."""
        logging.info('Modeling word clusters...')

        # get keywords
        logging.info('Getting keywords...')
        keyword_path = 'data/keywords.json'
        with open('data/keywords.json', 'r') as f:
            keywords = json.load(f)
        if len(keywords.strip()) == 0:
            keywords = self._paper_dataset.get_related_words()
            with open(keyword_path, 'w') as f:
                f.write(json.dumps(keywords, indent=4, ensure_ascii=False))

        # print the keywords
        from IPython import embed; embed()
        # get embeddings
        logging.info('Getting embeddings...')
        embeddings = self._encoder.encode(keywords)

        # clustering
        logging.info('Clustering...')
        clusters = KMeans(n_clusters=5, random_state=0).fit(embeddings)

        # visualize
        logging.info('Visualizing...')
        self._visualize(embedding=embeddings, labels=clusters.labels_)

        # TODO: make a logging file with the embeddings

    def model_topic_clusters(self) -> None:
        """Model the topics of papers by clustering."""

        # get papers

        pass

    def _visualize(self, embedding: np.ndarray, labels: List[int]) -> None:
        """Visualize the clustering results.

        Args:
            embedding   : the embedding to be visualized.
            labels      : the labels for the embedding.
        """
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            s=5,
            cmap='viridis'
            )
        plt.show()

    def model_cooccurring_keywords(self) -> None:
        """Model the co-occurring of the keyword and several selected words."""
        wordlist = ['machine learning', 'deep learning', 'neural networks', 'natural language processing', 'computer vision', 'reinforcement learning', 'robotics', 'ml', 'dl', 'nlp', 'cv', 'rl']

        pass

    def model_percent_numbers(self) -> None:
        """Model the percentage of numbers with respect to the year/conf."""
        # get data
        logging.info('Getting the percentage of numbers...')
        percent_numbers = self._paper_dataset.get_percent_numbers()

        # calculate an average of these numbers
        average = defaultdict(dict)
        for conf, papers in percent_numbers.items():
            for pid, fields in papers.items():
                year = fields['year'].split('_')[0]
                percent = fields['percent']
                if year not in average:
                    average[year] = 0
                average[year] += percent

        for year in average:
            paper_count = 0
            for conf, papers in percent_numbers.items():
                for pid, fields in papers.items():
                    if fields['year'].split('_')[0] == year:
                        paper_count += 1
            average[year] /= paper_count

        # plot the percentage 
        self._plot_float_features(
            percent_numbers=percent_numbers,
            average=average,
            save_path='assets/percent_numbers.png',
            y_label='Percentage of Numbers',
            title='Percentage of Numbers in Research Papers',
            colors= [
                '#8dd3c7',  # Light teal
                '#bebada',  # Light purple
                '#fb8072',  # Light salmon
                '#80b1d3',  # Light blue
                '#fdb462',  # Light orange
                '#b3de69',  # Light green
                '#fccde5',  # Light pink
                '#d9d9d9',  # Light gray
                '#bc80bd',  # Light violet
                '#ccebc5',  # Light mint
                '#ffed6f',  # Light yellow
                '#e41a1c',  # Strong red (for average)
                ]
            )

    def _plot_float_features(self,
                             percent_numbers: Dict[str, List[Dict[str, Any]]],
                             average: Dict[str, float] = None,
                             figsize: tuple = (12, 8),
                             save_path: str = None,
                             data_domain: str = 'percent',
                             y_label: str = None,
                             title: str = None,
                             colors: list = plt.cm.tab10.colors
                             ) -> None:
        """Plot a line chart showing the average feature value per year for each conference.

        Args:
            percent_number  : The conference data in the format {conference: [{year: year, percent: value}, ...]}.
            figsize         : Figure size as (width, height) in inches.
            save_path       : Path to save the figure.
            title           : Title for the plot.

        Returns:
            fig, ax         : The figure and axes objects for further customization if needed.
        """
        # process data
        logging.info('Processing data...')
        conference_years = defaultdict(lambda: defaultdict(list))
        all_years = set()

        for conference, papers in percent_numbers.items():
            for _, fields in papers.items():
                year = int(fields['year'].split('_')[0])
                percent = float(fields[data_domain])
                conference_years[conference][year].append(percent)
                all_years.add(year)

        # get averages by conference and year
        averages = {}
        for conference, years in conference_years.items():
            averages[conference] = {}
            for year, values in years.items():
                averages[conference][year] = np.mean(values)

        sorted_years = sorted(all_years)

        # create the plot
        logging.info('Creating the plot...')

        fig, ax = plt.subplots(figsize=figsize)

        for i, (conference, years_data) in enumerate(averages.items()):
            color = colors[i % len(colors)]

            xy_pairs = []
            for year in sorted_years:
                if year in years_data:
                    xy_pairs.append((year, years_data[year]))

            xy_pairs.sort(key=lambda pair: pair[0])

            x_values = [pair[0] for pair in xy_pairs]
            y_values = [pair[1] for pair in xy_pairs]

            ax.plot(x_values, y_values, marker='o', label=conference, color=color, 
                    linewidth=2, markersize=6, linestyle='-')

        if average:
            xy_pairs = []
            for year, value in average.items():
                xy_pairs.append((int(year), value))

            xy_pairs.sort(key=lambda pair: pair[0])

            x_values = [pair[0] for pair in xy_pairs]
            y_values = [pair[1] for pair in xy_pairs]

            ax.plot(x_values, y_values, marker='o', label='Average', color=colors[-1], 
                    linewidth=2, markersize=6, linestyle='-')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(y_label if y_label else 'Percentage of Numbers', fontsize=12)
        ax.set_title(title if title else 'Percentage of Numbers in Research Papers', fontsize=14)
    
        ax.set_xticks(sorted_years)
        ax.set_xticklabels(sorted_years, rotation=45)

        ax.grid(True, linestyle='--', alpha=0.7)

        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return

    def model_complexity_score(self) -> None:
        """Model the complexity trend of the research with respect to the year/conf."""
        logging.info('Modeling the complexity score...')

        # get complextity scores
        complexity_scores = self._paper_dataset.get_complexity_score()

        # calculate an average of these numbers
        average = defaultdict(dict)
        for conf, papers in complexity_scores.items():
            for pid, fields in papers.items():
                year = fields['year'].split('_')[0]
                score = fields['score']
                if year not in average:
                    average[year] = 0
                average[year] += score

        for year in average:
            paper_count = 0
            for conf, papers in complexity_scores.items():
                for pid, fields in papers.items():
                    if fields['year'].split('_')[0] == year:
                        paper_count += 1
            average[year] /= paper_count

        # plot
        self._plot_float_features(
            percent_numbers=complexity_scores,
            average=average,
            save_path='assets/complexity_scores.png',
            data_domain='score',
            y_label='Complexity Score',
            title='Complexity Score in Research Papers',
            colors=[
                '#8dd3c7',  # Light teal
                '#bebada',  # Light purple
                '#fb8072',  # Light salmon
                '#80b1d3',  # Light blue
                '#fdb462',  # Light orange
                '#b3de69',  # Light green
                '#fccde5',  # Light pink
                '#d9d9d9',  # Light gray
                '#bc80bd',  # Light violet
                '#ccebc5',  # Light mint
                '#ffed6f',  # Light yellow
                '#e41a1c',  # Strong red (for average)
            ]
            )

    def model_trend(self) -> None:
        """Model the trend of the research with respect to the year/conf."""
        logging.info('Modeling the trend grouping by conference...')

        # get papers by conference
        papers = self._paper_dataset.filter_papers()

        # build number diction
        number_dict = defaultdict(dict)
        for conf, conf_paper in papers.items():
            for fields in conf_paper:
                year = fields['year'].split('_')[0]
                if year not in number_dict[conf]:
                    number_dict[conf][year] = 0
                number_dict[conf][year] += 1

        # average
        average = defaultdict(int)
        for conf, years in number_dict.items():
            for year, number in years.items():
                if year not in average:
                    average[year] = 0
                average[year] += number
        for year in average:
            year_total = 0
            for conf, years in number_dict.items():
                if year in years:
                    year_total += 1
            average[year] /= year_total

        # plot the trend
        self._plot_trend(data=number_dict,
                         save_path='assets/trend.png',
                         average=average,
                         colors=[
                                '#8dd3c7',  # Light teal
                                '#bebada',  # Light purple
                                '#fb8072',  # Light salmon
                                '#80b1d3',  # Light blue
                                '#fdb462',  # Light orange
                                '#b3de69',  # Light green
                                '#fccde5',  # Light pink
                                '#d9d9d9',  # Light gray
                                '#bc80bd',  # Light violet
                                '#ccebc5',  # Light mint
                                '#ffed6f',  # Light yellow
                                '#e41a1c',  # Strong red (for average)
                            ]
                         )

    def _plot_trend(self,
                    data: Dict[str, List[Dict[str, Any]]],
                    save_path: str,
                    average: Dict[str, float] = None,
                    colors: list = plt.cm.tab10.colors
                    ) -> None:
        """Plot the trend of the research with respect to the year/conf.

        Args:
            data        : The conference data in the format {conference: [{year: year, percent: value}, ...]}.
            save_path   : Path to save the figure.
        """
        # process data
        logging.info('Processing data...')
        all_years = set()

        for conf, years in data.items():
            for year, number in years.items():
                all_years.add(year)

        numeric_years = [int(year) for year in all_years]
        sorted_years = sorted(numeric_years)

        # create the plot
        logging.info('Creating the plot...')
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (conference, years_data) in enumerate(data.items()):
            color = colors[i % len(colors)]

            xy_pairs = []
            for year in sorted_years:
                if str(year) in years_data:
                    xy_pairs.append((int(year), years_data[str(year)]))

            xy_pairs.sort(key=lambda pair: pair[0])

            x_values = [pair[0] for pair in xy_pairs]
            y_values = [pair[1] for pair in xy_pairs]

            ax.plot(x_values, y_values, marker='o', label=conference, color=color, 
                    linewidth=2, markersize=6, linestyle='-')

        if average:
            xy_pairs = []
            for year, value in average.items():
                xy_pairs.append((int(year), value))

            xy_pairs.sort(key=lambda pair: pair[0])

            x_values = [pair[0] for pair in xy_pairs]
            y_values = [pair[1] for pair in xy_pairs]

            ax.plot(x_values, y_values, marker='o', label='Average', color=colors[-1], 
                    linewidth=2, markersize=6, linestyle='-')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title('Trend of Research Papers', fontsize=14)

        ax.set_xticks(sorted_years)
        ax.set_xticklabels([str(year) for year in sorted_years], rotation=45)

        ax.grid(True, linestyle='--', alpha=0.7)

        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return

    def model_semantic_change(self) -> None:
        """Model the semantic change of the research with respect to the year.
        """
        # get papers by year and conference
        papers = self._paper_dataset.filter_papers()

        # seperate the papers by decade
        decades = defaultdict(lambda: defaultdict(list))
        for conf, papers in papers.items():
            for pid, fields in papers.items():
                year = fields['year'].split('_')[0]
                decade = year[:3] + '0'
                decades[decade][conf].append(fields['text'])

        # main loop
        # TODO: rethink embedding's structure
        embeddings = defaultdict(list)

        for decade in decades:
            # get the papers
            papers = decades[decade]

            # load or train the word2vec model
            logging.info('Loading or training the Word2Vec model...')
            model_path = f'data/word2vec_{decade}.model'
            if model_path.exists():
                model = Word2Vec.load(model_path)
            else:
                self._train_word2vec(
                    model_path=model_path,
                    data=papers
                    )

            # get the embeddings of the keywords (input and concerned)
            anchor_words = [] # TODO: get from the keywords, should be loaded

            words_to_check = anchor_words.append(self._keyword)

            for word in words_to_check:
                if word in model.wv:
                    embeddings[decade].append(model.wv[word])
                else:
                    logging.warning(f"'{word}' not in vocabulary")


        pass

    def _train_word2vec(self,
                        model_path: str,
                        data: List[List[str]],
                        vector_size: int = 100,
                        window: int = 5,
                        min_count: int = 1,
                        workers: int = 4,
                        sg: int = 1,
                        epochs: int = 100
                        ) -> None:
        """Train the Word2Vec model."""
        logging.info('Training the Word2Vec model...')

        # train model
        model = Word2Vec(
            sentences=data,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            epochs=epochs
        )

        # save the model
        model.save(model_path)

        return

