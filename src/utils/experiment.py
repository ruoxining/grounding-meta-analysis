"""Class to conduct experiments."""
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
        keywords = self._paper_dataset.get_related_words()

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

    def model_percent_numbers(self,
                              by: str = 'year',
                              ) -> None:
        """Model the percentage of numbers with respect to the year/conf.

        Args:
            by  : the category to model the trend by, 'year', 'conf'.
        """
        pass

    def model_complexity_score(self,
                               by: str = 'year',
                               ) -> None:
        """Model the complexity trend of the research with respect to the year/conf.

        Args:
            by  : the category to model the trend by, 'year', 'conf'.
        """
        pass

    def model_trend(self,
                          by: str = 'year',
                          ) -> None:
        """Model the trend of the research with respect to the year/conf.

        Args:
            by  : the category to model the trend by, 'year', 'conf'.
        """
        # get papers
        papers = self._paper_dataset.group_papers(by=by)

        # get counts
        counts = defaultdict(int)
        for category, papers in papers.items():
            counts[category] = len(papers)

        # plot
        plt.bar(counts.keys(), counts.values())
        plt.show()
