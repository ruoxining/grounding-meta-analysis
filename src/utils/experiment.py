"""Class to conduct experiments."""
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
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
        # NOTE: high

        # get data
        logging.info('Getting the percentage of numbers...')
        percent_numbers = self._paper_dataset.get_percent_numbers()

        # plot the percentage 
        self._plot_percent_numbers(percent_numbers=percent_numbers, save_path='assets/percent_numbers.png')

    def _plot_percent_numbers(self,
                              percent_numbers: Dict[str, List[Dict[str, Any]]],
                              figsize: tuple = (12, 8),
                              save_path: str = None,
                              title: str = None
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
            for pid, fields in papers.items():
                year = int(fields['year'].split('_')[0])
                percent = float(fields['percent'])
                conference_years[conference][year].append(percent)
                all_years.add(year)

        # get averages
        averages = {}
        for conference, years in conference_years.items():
            averages[conference] = {}
            for year, values in years.items():
                averages[conference][year] = np.mean(values)

        sorted_years = sorted(all_years)

        # create the plot
        logging.info('Creating the plot...')

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10.colors

        for i, (conference, years_data) in enumerate(averages.items()):
            color = colors[i % len(colors)]

            x_data = sorted_years
            y_data = [years_data.get(year, np.nan) for year in sorted_years]

            ax.plot(x_data, y_data, marker='o', label=conference, color=color, 
                linewidth=2, markersize=6, linestyle='-')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Percentage of Numbers', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Percentage of Numbers in Research Papers', fontsize=14)
    
        ax.set_xticks(sorted_years)
        ax.set_xticklabels(sorted_years, rotation=45)

        ax.grid(True, linestyle='--', alpha=0.7)

        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return

    def model_complexity_score(self,
                               by: str = 'year',
                               ) -> None:
        """Model the complexity trend of the research with respect to the year/conf.

        Args:
            by  : the category to model the trend by, 'year', 'conf'.
        """
        # NOTE: high
        pass

    def model_trend(self,
                          by: str = 'year',
                          ) -> None:
        """Model the trend of the research with respect to the year/conf.

        Args:
            by  : the category to model the trend by, 'year', 'conf'.
        """
        # NOTE: high

        # get papers
        papers = self._paper_dataset.group_papers(by=by)

        # get counts
        counts = defaultdict(int)
        for category, papers in papers.items():
            counts[category] = len(papers)

        # plot
        plt.bar(counts.keys(), counts.values())
        plt.show()

    def model_semantic_change(self) -> None:
        """Model the semantic change of the research with respect to the year.
        """
        # NOTE: high
        pass
