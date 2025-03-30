"""Class for the experiment to detect the semantic change."""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .dataset import PaperDataset


class ExprSemanticChange:
    """Class for the experiment to detect the semantic change."""

    def __init__(self, keyword, api):
        """Initialize the class.

        Args:
            keyword: The keyword to track for semantic change
            api: The API instance for accessing data
            paper_dataset: Optional dataset of papers
        """
        self._keyword = keyword  # keeping both for compatibility
        self._api = api
        self._paper_dataset = PaperDataset(keyword=keyword)
        self._anchor_words = ['model', 'information', 'feature', 'system', 'concept', 'course', 'computer', 'vision', 'symbol', 'robot', 'language', 'algorithm', 'robotics', 'AI', 'ML', 'pattern', 'recognition', 'learning', 'neural', 'network', 'modality', 'speech', 'visual', 'CV', 'symbolic', 'physic', 'cognition', 'understand', 'reason', 'knowledge', 'representation', 'perception', 'action', 'planning', 'control', 'decision', 'making', 'processing', 'NLP', 'generation', 'translation', 'dialogue', 'chatbot', 'reasoning', 'logic', 'inference', 'explainable', 'interpretable', 'explain', 'interpret']

    def model_semantic_change(self) -> None:
        """Model the semantic change of the research with respect to the year.
        """
        # get the embeddings
        embeddings = self._get_embeddings()

        # match the embeddings
        self._match_embeddings(embeddings=embeddings)

        # calculate the semantic change
        semantic_changes = self._calculate_semantic_change(embeddings=embeddings)

        # get the embeddings of the keywords (input and concerned)
        words_to_plot = self._anchor_words + [self._keyword]

        # plot the semantic change
        self._plot_semantic_change(embeddings=embeddings,
                                   words_to_plot=words_to_plot,
                                   semantic_changes=semantic_changes
                                   )

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
        model = gensim.models.Word2Vec(
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

    def _get_embeddings(self) -> Dict[str, Dict[str, List[float]]]:
        """Get the embeddings through training or loading.

        Returns:
            Dictionary of decades with words and their embeddings
        """
        logging.info('Getting embeddings...')

        # TODO: now the decades are hard coded, should find a way to get them dynamically, e.g., generating and loading
        decades_all = ['1980', '1990', '2000', '2010', '2020']

        # check if embeddings already exist
        # TODO
        decades_found = []
        embeddings = defaultdict(dict) # embeddings: conf: [key: embedding]

        for filename in os.listdir('data'):
            if filename.endswith('_embeddings.json'):
                decade = filename.split('_')[0]
                decades_found.append(decade)
                data_path = f'data/{decade}_embeddings.json'
                with open(data_path, 'r') as f:
                    embeddings[decade] = json.load(f)
                continue

        if set(decades_all) == set(decades_found):
            logging.info('Embeddings found, skipping training...')
            return embeddings

        logging.info('Embeddings not found, training...')
        decades_todo = list(set(decades_all) - set(decades_found))

        # get papers by year and conference
        papers = self._paper_dataset.dataset

        # seperate the papers by decade
        decades = defaultdict(lambda: defaultdict(list))
        for conf, conf_papers in papers.items():
            for fields in conf_papers:
                year = fields['year'].split('_')[0]
                decade = year[:3] + '0'
                if decade in decades_todo:
                    decades[decade][conf].append(fields['text'])

        # to print the decade's key
        # main loop
        for decade in decades.keys():
            # get the papers
            train_data = []
            for conf, texts in decades[decade].items():
                for text in texts:
                    text_data = []
                    clean_text = text.replace('\n', ' ')
                    for sent in nltk.tokenize.sent_tokenize(clean_text):
                        text_data.append([word.lower() for word in nltk.tokenize.word_tokenize(sent)])
                    train_data.extend(text_data)                

            # load or train the word2vec model
            logging.info('Loading or training the Word2Vec model...')
            model_path = f'models/word2vec_{decade}.model'
            if os.path.exists(model_path):
                model = gensim.models.Word2Vec.load(model_path)
            else:
                self._train_word2vec(
                    model_path=model_path,
                    data=train_data
                    )
                model = gensim.models.Word2Vec.load(model_path)

            # build the vocabulary
            vocabulary = set()
            for conf, texts in decades[decade].items():
                for text in texts:
                    for sent in nltk.tokenize.sent_tokenize(text):
                        for word in nltk.tokenize.word_tokenize(sent):
                            vocabulary.add(word)

            # save the embeddings
            embeddings[decade] = defaultdict(list)
            for word in vocabulary:
                if word in model.wv:
                    embeddings[decade][word] = model.wv[word].tolist()
                else:
                    logging.warning(f"'{word}' not in vocabulary")

            with open(f'data/{decade}_embeddings.json', 'w') as f:
                f.write(json.dumps(embeddings[decade], indent=4, ensure_ascii=False))

        return embeddings

    def _match_embeddings(self,
                          embeddings: Dict[str, Dict[str, List[float]]],
                          ) -> Dict[str, np.ndarray]:
        """Match the embeddings of the keywords by training and transforming.

        Args:
            embeddings: The embeddings of the keywords, shape decade: word: embedding.
            
        Returns:
            Dictionary of transformation matrices by decade
        """
        logging.info('Matching the embeddings...')

        all_decades = sorted(list(embeddings.keys()))
        base_decade = all_decades[-1]  # Using the latest decade as base
        transformation_matrices = {}
        
        # ensure output directories exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # iterate through all decades (except the base decade)
        for decade in all_decades[:-1]:
            logging.info(f'Matching embeddings for {decade}...')

            # find common words between this decade and base decade
            common_words = set(embeddings[decade].keys()) & set(embeddings[base_decade].keys())
            logging.info(f'Found {len(common_words)} common words between {decade} and {base_decade}')
            
            if len(common_words) < 10:
                logging.warning(f'Too few common words between {decade} and {base_decade}, skipping')
                continue
            
            # extract embeddings for common words
            decade_embed = np.array([embeddings[decade][word] for word in common_words])
            base_embed = np.array([embeddings[base_decade][word] for word in common_words])
            
            # match the embeddings
                # match the embeddings
            W = self._match_embeddings_pair(decade_embed, base_embed, epochs=1000, learning_rate=0.01)
            transformation_matrices[decade] = W

            # evaluate the alignment
            score = self._evaluate_alignment(decade_embed, base_embed, W)
            logging.info(f'Alignment score: {score}')

            # transform all embeddings from this decade to base decade space
            all_decade_embeds = np.array(list(embeddings[decade].values()))
            transformed = self._transform_vectors(all_decade_embeds, W)
            
            # save the transformed embeddings
            transformed_embeddings = dict(zip(embeddings[decade].keys(), 
                                             [vec.tolist() for vec in transformed]))
            
            with open(f'data/{decade}_transformed_embeddings.json', 'w') as f:
                f.write(json.dumps(transformed_embeddings, indent=4, ensure_ascii=False))
            
            # save the transformation matrix
            np.save(f'models/transformation_matrix_{decade}_to_{base_decade}.npy', W)

        return transformation_matrices

    def _match_embeddings_pair(self, 
                               src_embeddings: np.ndarray,
                               tgt_embeddings: np.ndarray,
                               epochs: int = 1000,
                               learning_rate: float = 0.01) -> np.ndarray:
        """Match the embeddings of the keywords using gradient descent.

        Args:
            src_embeddings: The source embeddings, shape (n_words, dimension).
            tgt_embeddings: The target embeddings, shape (n_words, dimension).
            epochs: Number of epochs for training.
            learning_rate: Learning rate for gradient descent.

        Returns:
            W: The trained transformation matrix.
        """
        # randomly initialize the transformation matrix W
        np.random.seed(42)
        W = np.random.rand(src_embeddings.shape[1], src_embeddings.shape[1])

        for epoch in range(epochs):
            # forward pass: compute predicted Y
            predicted = src_embeddings @ W

            # compute and print loss
            loss = np.mean((predicted - tgt_embeddings) ** 2)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            # backward pass: compute gradients
            grad_w = 2 / src_embeddings.shape[0] * src_embeddings.T @ (predicted - tgt_embeddings)

            # update weights
            W -= learning_rate * grad_w

        return W

    def _transform_vectors(self, 
                           vectors: np.ndarray, 
                           W: np.ndarray
                           ) -> np.ndarray:
        """Apply the transformation matrix to vectors.

        Args:
            vectors: The vectors to be transformed, shape (n_words, dimension).
            W: The transformation matrix, shape (dimension, dimension).

        Returns:
            transformed: The transformed vectors.
        """
        return vectors @ W

    def _evaluate_alignment(self,
                            src_embeddings: np.ndarray,
                            tgt_embeddings: np.ndarray,
                            W: np.ndarray,
                            top_k: int = 10
                            ) -> float:
        """Evaluate the alignment of the embeddings.

        Args:
            src_embeddings: The source embeddings, shape (n_words, dimension).
            tgt_embeddings: The target embeddings, shape (n_words, dimension).
            W: The transformation matrix, shape (dimension, dimension).
            top_k: Number of top similarities to consider.

        Returns:
            score: The alignment score.
        """
        # transform the source embeddings
        transformed = self._transform_vectors(src_embeddings, W)

        # normalize the embeddings
        transformed = normalize(transformed, axis=1)
        tgt_embeddings = normalize(tgt_embeddings, axis=1)

        # calculate the cosine similarity
        similarities = transformed @ tgt_embeddings.T
        
        # get diagonal elements (similarities between corresponding vectors)
        direct_similarities = np.diag(similarities)

        # return average similarity
        return np.mean(direct_similarities)

    def _calculate_semantic_change(self,
                                   embeddings: Dict[str, Dict[str, List[float]]]
                                   ) -> Dict[str, Dict[str, float]]:
        """Calculate the semantic change.
        
        Args:
            embeddings: Dictionary of decades with words and their embeddings
            
        Returns:
            Dictionary of decades with words and their semantic change scores
        """
        logging.info('Calculating semantic change...')

        all_decades = sorted(list(embeddings.keys()))
        base_decade = all_decades[-1]  # Using the latest decade as reference
        semantic_changes = {}

        # load transformed embeddings for each decade
        transformed_embeddings = {}
        for decade in all_decades[:-1]:  # Skip the base decade
            try:
                with open(f'data/{decade}_transformed_embeddings.json', 'r') as f:
                    transformed_embeddings[decade] = json.load(f)
            except FileNotFoundError:
                logging.warning(f'Transformed embeddings for {decade} not found, skipping')
                continue

        # add the base decade embeddings
        transformed_embeddings[base_decade] = embeddings[base_decade]

        # calculate semantic change for each decade relative to base
        for decade in all_decades[:-1]:
            if decade not in transformed_embeddings:
                continue

            semantic_changes[decade] = {}

            # find common words between this decade and base decade
            common_words = set(transformed_embeddings[decade].keys()) & set(transformed_embeddings[base_decade].keys())

            for word in common_words:
                # get embeddings
                vec1 = np.array(transformed_embeddings[decade][word])
                vec2 = np.array(transformed_embeddings[base_decade][word])

                # normalize vectors
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = vec2 / np.linalg.norm(vec2)

                # calculate cosine similarity (1 - cosine distance)
                similarity = np.dot(vec1, vec2)

                # semantic change is the inverse of similarity (higher = more change)
                semantic_changes[decade][word] = 1.0 - similarity

        # save semantic changes
        with open('data/semantic_changes.json', 'w') as f:
            f.write(json.dumps(semantic_changes, indent=4, ensure_ascii=False))
            
        return semantic_changes

    def _plot_semantic_change(self,
                              embeddings: Dict[str, Dict[str, List[float]]],
                              ) -> None:
        """Plot the semantic change.
        
        Args:
            embeddings: Dictionary of decades with words and their embeddings
        """
        logging.info('Plotting semantic change...')

        all_decades = sorted(list(embeddings.keys()))

        self._plot_trajectory(embeddings, all_decades)

    def _plot_trajectory(self,
                         embeddings: Dict[str, Dict[str, List[float]]],
                         decades: List[str]) -> None:
        """Plot the trajectory of words in 2D space.
        
        Args:
            embeddings: Dictionary of decades with words and their embeddings
            words_to_plot: List of words to plot
            decades: List of decades in chronological order
        """
        # load transformed embeddings
        transformed_embeddings = {}
        for decade in decades[:-1]:  # Skip the base decade
            try:
                with open(f'data/{decade}_transformed_embeddings.json', 'r') as f:
                    transformed_embeddings[decade] = json.load(f)
            except FileNotFoundError:
                logging.warning(f'Transformed embeddings for {decade} not found, skipping')
                continue
        
        # add the base decade embeddings
        transformed_embeddings[decades[-1]] = embeddings[decades[-1]]

        # extract keyword (first element) and anchor words (rest of elements)
        keyword = self._keyword

        # collect embeddings for keyword across decades
        word_trajectories = defaultdict(list)
        decade_labels = []

        # get trajectory for keyword across all decades
        for decade in sorted(transformed_embeddings.keys()):
            if keyword in transformed_embeddings[decade]:
                word_trajectories[keyword].append(transformed_embeddings[decade][keyword])
            decade_labels.append(decade)

        # check if we have enough data for keyword
        if len(word_trajectories[keyword]) < 2:
            logging.warning(f'Not enough data to plot trajectory for keyword "{keyword}"')
            return

        # collect anchor word embeddings (only from the latest decade)
        anchor_embeddings = {}
        latest_decade = decades[-1]

        for word in self._anchor_words:
            if word in transformed_embeddings[latest_decade]:
                anchor_embeddings[word] = transformed_embeddings[latest_decade][word]
            else:
                logging.warning(f'Anchor word "{word}" not found in latest decade')

        # get all embeddings for PCA
        all_embeddings = []
        # add keyword trajectory
        all_embeddings.extend(word_trajectories[keyword])
        # add anchor points
        all_embeddings.extend(list(anchor_embeddings.values()))

        if not all_embeddings:
            logging.warning('No embeddings to plot')
            return

        # apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        pca.fit(all_embeddings)

        # create figure
        plt.figure(figsize=(12, 8))

        # plot keyword trajectory
        if len(word_trajectories[keyword]) >= 2:
            # project to 2D
            trajectory = pca.transform(word_trajectories[keyword])

            # plot line with larger markers for keyword
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', 
                     label=keyword, linewidth=2, markersize=8)

            # annotate decades for keyword
            for i, decade in enumerate(decade_labels[:len(trajectory)]):
                plt.annotate(decade, (trajectory[i, 0], trajectory[i, 1]), 
                             textcoords="offset points", xytext=(0, 10), ha='center')

        # plot anchor words as single points
        for word, embedding in anchor_embeddings.items():
            # project to 2D
            point = pca.transform([embedding])[0]

            # plot as single point with different marker
            plt.scatter(point[0], point[1], marker='*', s=100, label=word)

            # annotate the word
            plt.annotate(word, (point[0], point[1]), 
                         textcoords="offset points", xytext=(5, 5), ha='left')

        plt.title(f'Semantic Change Trajectory for "{keyword}"')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('assets/semantic_trajectory.pdf', dpi=300)
        plt.close()
