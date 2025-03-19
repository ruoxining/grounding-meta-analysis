"""Class to conduct experiments."""
import logging
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
    def __init__(self, keyword: str) -> None:
        """Initialize the Experiments class."""
        self._paper_dataset = PaperDataset(keyword=keyword)
        self._querier = Querier(model_name='gpt-4', api_key='api_key')
        self._encoder = SentenceTransformer('all-mpnet-base-v2')

    def cluster_words(self,
                      type_keyword: str = 'keyword',
                      ) -> None:
        """Cluster the keywords in the papers.

        Args:
            type    : the type of keywords to cluster, 'keyword', 'topic'.
        """
        # get the keywords
        logging.info(f"Clustering the keywords in the papers with type {type_keyword}.")
        if type_keyword == 'keyword':
            keywords = self._paper_dataset.get_related_words()
            # TODO: arrange format of the keywords
        elif type_keyword == 'topic':
            keywords = self._paper_dataset.get_topics()
            # TODO: arrange format of the keywords
        else:
            raise ValueError("Invalid type.")

        # get the embeddings
        embeddings = self._encoder.encode(keywords)

        # clustering
        clusters = KMeans(n_clusters=5, random_state=0).fit(embeddings)

        # visualize
        self._visualize(embedding=embeddings, labels=clusters.labels_)

    def _clustering(self, embeddings: List[torch.Tensor]) -> None:
        """Perform clustering on input embeddings.

        Args:
            embeddings  : the embeddings to be clustered.
        """
        pass

    def _reduce(self,
                   features: List[torch.Tensor],
                   toolkit: str = 'umap',
                   n_neighbors: int = 15,
                   min_dist: float = 0.1,
                   n_components: int = 2,
                   metric: str = 'euclidean'
                   ) -> None:
        """Visualize the clustering results.

        Args:
            features    : the features to be visualized.s
            toolkit     : the toolkit to use for visualization, default 'umap', choices: ['umap'].
            n_neighbors : the number of neighbors to consider.
            min_dist    : the minimum distance between points.
            n_components: the number of components.
            metric      : the metric to use.
        """
        if toolkit == 'umap':
            import umap
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
            embedding = reducer.fit_transform(features)
        else:
            raise ValueError("Invalid toolkit.")

    def _visualize(self, embedding: np.ndarray, labels: List[int]) -> None:
        """Visualize the clustering results.

        Args:
            embedding   : the embedding to be visualized.
            labels      : the labels for the embedding.
        """
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=5, cmap='viridis')
        plt.show()

    def model_trend_graph(self,
                          by: str = 'year',
                          ) -> None:
        """Model the trend of the research with respect to the year/conf.

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

    def model_percent_numbers(self,
                              by: str = 'year',
                              ) -> None:
        """Model the percentage of numbers with respect to the year/conf.

        Args:
            by  : the category to model the trend by, 'year', 'conf'.
        """
        pass
