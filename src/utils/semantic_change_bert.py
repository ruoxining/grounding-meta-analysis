"""Calculate the semantic change with BERT."""
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .semantic_change import ExprSemanticChange


class ExprSemanticChangeBERT(ExprSemanticChange):
    """Class for semantic change with BERT model."""

    def __init__(self, keyword, api):
        """Initialize the class."""
        super().__init__(keyword, api)
        self._decades_all = ["1980", "1990", "2000", "2010", "2020"]
        self._anchor_words = ['embodiment', 'context', 'symbol', 'reality', 'physical', 'meaning', 'cognition', 'robotics', 'vision', 'understand']

    def model_semantic_change(self):
        """Model the semantic change by querying a BERT model."""

        # load the bert model
        self._init_encoder()

        # get the embedings
        embeddings = self._get_embeddings()

        # get anchor word embeddings
        anchor_embeddings = self._get_anchor_embeddings()

        # plot the results
        self._plot_trajectory(
            embeddings=embeddings,
            decades=self._decades_all,
            anchor_embeddings=anchor_embeddings
        )

    def _init_encoder(self):
        """Initialize the encoder."""
        logging.info("initializing the encoder.")

        self._tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self._model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    def _get_embeddings(self) -> Dict[str, Dict[str, dict]]:
        """get the embeddings of the keyword from the encoder."""
        logging.info("getting the embeddings...")

        # check if embeddings exist
        decades_todo = []
        embeddings = defaultdict(dict)

        for decade in self._decades_all:
            file_path = f"data/{decade}_embeddings_bert.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    embeddings[decade] = json.load(f)
                    logging.info(f"loaded embeddings for decade {decade} from {file_path}")
            else:
                decades_todo.append(decade)

        if not decades_todo:
            logging.info("all embeddings already exist.")
            return embeddings

        logging.info(f"computing embeddings for decades: {decades_todo}")
        
        # get papers, only from abstracts where the keyword appears in abstract
        papers = self._paper_dataset.filter_papers(in_full_text=False, get_section='abstract')

        # separate the papers by decade
        decades = defaultdict(lambda: defaultdict(list))
        for conf, conf_papers in papers.items():
            for fields in conf_papers:
                year = fields['year'].split('_')[0]
                decade = year[:3] + '0'
                if decade in decades_todo:
                    decades[decade][conf].append(fields)

        # filter sentences with the keyword
        keyword = self._keyword.lower()
        
        # initialize the encoder if not already done
        if not hasattr(self, '_model') or not hasattr(self, '_tokenizer'):
            self._init_encoder()
        
        # process each decade
        for decade, conferences in decades.items():
            decade_embeddings = {}
            paper_count = 0

            for conf, papers_data in conferences.items():
                for i, paper in enumerate(papers_data):
                    paper_text = paper['text']
                    
                    # filter sentences containing the keyword from abstract
                    sentences = [s.strip() for s in paper_text.split('.') if s.strip()]
                    relevant_sentences = [s for s in sentences if keyword in s.lower()]
                    
                    if not relevant_sentences:
                        continue
                    
                    paper_embeddings = []
                    
                    # get embeddings for each relevant sentence
                    for sentence in relevant_sentences:
                        try:
                            # tokenize the sentence with explicit max length
                            inputs = self._tokenizer(
                                sentence, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True,
                                max_length=512  # explicitly set max length
                            )
                            
                            # get the model outputs
                            with torch.no_grad():
                                outputs = self._model(**inputs)
                            
                            # get the last hidden state
                            last_hidden_state = outputs.last_hidden_state
                            
                            # find the tokens for the keyword
                            input_ids = inputs.input_ids[0].tolist()
                            tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
                            
                            # tokenize the keyword to find its subwords
                            keyword_tokens = self._tokenizer.tokenize(keyword)
                            
                            # find positions of the keyword tokens
                            keyword_positions = []
                            for j in range(len(tokens) - len(keyword_tokens) + 1):
                                if tokens[j:j+len(keyword_tokens)] == keyword_tokens:
                                    keyword_positions.extend(range(j, j+len(keyword_tokens)))
                            
                            if not keyword_positions:
                                continue
                            
                            # average the embeddings of the keyword tokens
                            keyword_embeddings = last_hidden_state[0, keyword_positions].mean(dim=0)
                            paper_embeddings.append(keyword_embeddings.cpu().numpy())
                        except Exception as e:
                            logging.warning(f"error processing sentence: {e}")
                            continue
                    
                    if paper_embeddings:
                        # average embeddings for the paper
                        avg_embedding = np.mean(paper_embeddings, axis=0)
                        paper_id = f"{conf}_{paper.get('index', i)}"  # use index if available, otherwise use iterator
                        decade_embeddings[paper_id] = {
                            'embedding': avg_embedding.tolist(),
                            'year': paper['year']
                        }
                        paper_count += 1
            
            logging.info(f"processed {paper_count} papers for decade {decade}")
            
            if paper_count > 0:
                # save embeddings for this decade
                file_path = f"data/{decade}_embeddings_bert.json"
                with open(file_path, 'w') as f:
                    json.dump(decade_embeddings, f)
                
                logging.info(f"saved embeddings for decade {decade} to {file_path}")
                
                embeddings[decade] = decade_embeddings
            else:
                logging.warning(f"no embeddings found for decade {decade}")

        return embeddings
    
    def _get_anchor_embeddings(self) -> Dict[str, np.ndarray]:
        """get embeddings for anchor words."""
        logging.info("getting anchor word embeddings...")
        
        # check if anchor embeddings exist
        anchor_path = f"data/anchor_embeddings_bert.json"
        if os.path.exists(anchor_path):
            with open(anchor_path, 'r') as f:
                anchor_data = json.load(f)
                anchor_embeddings = {word: np.array(emb) for word, emb in anchor_data.items()}
                logging.info(f"loaded anchor embeddings from {anchor_path}")
                return anchor_embeddings
        
        # initialize embeddings dictionary
        anchor_embeddings = {}
        
        # ensure model is initialized
        if not hasattr(self, '_model') or not hasattr(self, '_tokenizer'):
            self._init_encoder()
        
        # get embeddings for each anchor word
        for word in self._anchor_words:
            try:
                # encode the word
                inputs = self._tokenizer(
                    word,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # get the model outputs
                with torch.no_grad():
                    outputs = self._model(**inputs)
                
                # get the last hidden state and use [CLS] token or average of subword tokens
                word_tokens = self._tokenizer.tokenize(word)
                
                if len(word_tokens) > 1:
                    # for multi-token words, average over all subword tokens
                    token_embeddings = []
                    for i in range(len(word_tokens)):
                        token_embeddings.append(outputs.last_hidden_state[0, i+1].cpu().numpy())  # +1 to skip [CLS]
                    word_embedding = np.mean(token_embeddings, axis=0)
                else:
                    # for single token words, use the token embedding directly
                    word_embedding = outputs.last_hidden_state[0, 1].cpu().numpy()  # index 1 to skip [CLS]
                
                anchor_embeddings[word] = word_embedding
                logging.info(f"generated embedding for anchor word: {word}")
            except Exception as e:
                logging.warning(f"error processing anchor word {word}: {e}")
        
        # save anchor embeddings
        with open(anchor_path, 'w') as f:
            anchor_data = {word: emb.tolist() for word, emb in anchor_embeddings.items()}
            json.dump(anchor_data, f)
        
        logging.info(f"saved anchor embeddings to {anchor_path}")
        return anchor_embeddings

    def _plot_trajectory(self, embeddings, decades, anchor_embeddings=None):
        """plot the trajectory of the keyword embeddings over decades using pca."""
        logging.info("plotting the trajectory using pca...")
        
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            logging.error("matplotlib and scikit-learn required for plotting.")
            return
        
        # prepare data for visualization
        all_embeddings = []
        embedding_decades = []
        embedding_ids = []
        
        for decade in decades:
            if decade not in embeddings or not embeddings[decade]:
                continue
                
            decade_data = embeddings[decade]
            for paper_id, data in decade_data.items():
                # convert to numpy if it's a list
                emb = np.array(data['embedding'])
                all_embeddings.append(emb)
                embedding_decades.append(decade)
                embedding_ids.append(paper_id)
        
        if not all_embeddings:
            logging.warning("no embeddings to plot.")
            return
            
        # convert to numpy array
        all_embeddings = np.array(all_embeddings)
        
        # add anchor embeddings to the mix for PCA
        combined_embeddings = all_embeddings.copy()
        if anchor_embeddings:
            anchor_array = np.array([emb for emb in anchor_embeddings.values()])
            combined_embeddings = np.vstack([combined_embeddings, anchor_array])
        
        # ensure output directory exists
        os.makedirs("plots", exist_ok=True)
        
        # apply pca for dimensionality reduction
        try:
            pca = PCA(n_components=2)
            vis_combined = pca.fit_transform(combined_embeddings)
            vis_embeddings = vis_combined[:len(all_embeddings)]
            
            # if we have anchor embeddings, get their projections
            if anchor_embeddings:
                vis_anchors = vis_combined[len(all_embeddings):]
            
            variance_explained = pca.explained_variance_ratio_
            logging.info(f"pca variance explained: {variance_explained[0]:.3f}, {variance_explained[1]:.3f}")
            logging.info(f"total variance explained: {sum(variance_explained):.3f}")
        except Exception as e:
            logging.error(f"pca failed: {e}")
            return
        
        # plot
        plt.figure(figsize=(10, 8))
        
        # define colors for decades
        decade_colors = {
            "1980": "blue",
            "1990": "green",
            "2000": "orange",
            "2010": "red",
            "2020": "purple"
        }
        
        # plot each decade
        for decade in sorted(set(embedding_decades)):
            decade_indices = [i for i, d in enumerate(embedding_decades) if d == decade]
            if decade_indices:  # check if we have any points for this decade
                plt.scatter(
                    vis_embeddings[decade_indices, 0],
                    vis_embeddings[decade_indices, 1],
                    label=f"{decade} (n={len(decade_indices)})",
                    color=decade_colors.get(decade, "gray"),
                    alpha=0.7
                )
        
        # plot anchor words with distinct colors
        if anchor_embeddings:
            anchor_words = list(anchor_embeddings.keys())
            # use a colormap to get distinct colors for anchor words
            from matplotlib.cm import get_cmap
            cmap = get_cmap('tab20')
            colors = [cmap(i/len(anchor_words)) for i in range(len(anchor_words))]
            
            for i, word in enumerate(anchor_words):
                plt.scatter(
                    vis_anchors[i, 0],
                    vis_anchors[i, 1],
                    label=f"Anchor: {word}",
                    color=colors[i],
                    marker='*',
                    s=150,
                    edgecolors='black'
                )
                
                # add text labels to anchor words
                plt.annotate(
                    word,
                    (vis_anchors[i, 0], vis_anchors[i, 1]),
                    fontsize=9,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        plt.title(f"semantic trajectory of '{self._keyword}' over decades (pca)")
        plt.xlabel(f"pc1 ({variance_explained[0]:.1%} variance)")
        plt.ylabel(f"pc2 ({variance_explained[1]:.1%} variance)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # save the plot
        plt.savefig(f"plots/{self._keyword}_semantic_trajectory_pca.pdf", dpi=300, bbox_inches="tight")
        logging.info(f"plot saved to plots/{self._keyword}_semantic_trajectory_pca.pdf")
        
        # also save a version without legend for better viewing of the plot itself
        plt.figure(figsize=(10, 8))
        
        # plot each decade
        for decade in sorted(set(embedding_decades)):
            decade_indices = [i for i, d in enumerate(embedding_decades) if d == decade]
            if decade_indices:
                plt.scatter(
                    vis_embeddings[decade_indices, 0],
                    vis_embeddings[decade_indices, 1],
                    color=decade_colors.get(decade, "gray"),
                    alpha=0.7
                )
        
        # plot anchor words
        if anchor_embeddings:
            anchor_words = list(anchor_embeddings.keys())
            colors = [cmap(i/len(anchor_words)) for i in range(len(anchor_words))]
            
            for i, word in enumerate(anchor_words):
                plt.scatter(
                    vis_anchors[i, 0],
                    vis_anchors[i, 1],
                    color=colors[i],
                    marker='*',
                    s=150,
                    edgecolors='black'
                )
                
                # add text labels to anchor words
                plt.annotate(
                    word,
                    (vis_anchors[i, 0], vis_anchors[i, 1]),
                    fontsize=9,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        plt.title(f"semantic trajectory of '{self._keyword}' (no legend)")
        plt.xlabel(f"pc1 ({variance_explained[0]:.1%} variance)")
        plt.ylabel(f"pc2 ({variance_explained[1]:.1%} variance)")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # save the plot without legend
        plt.savefig(f"plots/{self._keyword}_semantic_trajectory_pca_no_legend.pdf", dpi=300, bbox_inches="tight")
        
        # calculate and plot the centroid for each decade
        decades_with_data = sorted([d for d in set(embedding_decades) if embedding_decades.count(d) >= 3])

        if len(decades_with_data) > 1:
            plt.figure(figsize=(10, 8))

            centroids = {}
            for decade in decades_with_data:
                decade_indices = [i for i, d in enumerate(embedding_decades) if d == decade]
                if len(decade_indices) >= 3:  # only calculate centroids for decades with enough data
                    centroid = np.mean(vis_embeddings[decade_indices], axis=0)
                    centroids[decade] = centroid
                    
                    plt.scatter(
                        centroid[0],
                        centroid[1],
                        label=decade,
                        color=decade_colors.get(decade, "gray"),
                        s=100,
                        edgecolors="black"
                    )

            # connect centroids with arrows to show direction of change
            decades_sorted = sorted(centroids.keys())
            for i in range(len(decades_sorted) - 1):
                current = decades_sorted[i]
                next_decade = decades_sorted[i + 1]
                
                # skip if there's no clear direction (too close)
                distance = np.linalg.norm(centroids[next_decade] - centroids[current])
                if distance < 0.01:
                    continue

                plt.arrow(
                    centroids[current][0],
                    centroids[current][1],
                    centroids[next_decade][0] - centroids[current][0],
                    centroids[next_decade][1] - centroids[current][1],
                    head_width=0.5,
                    head_length=0.5,
                    fc=decade_colors.get(next_decade, "gray"),
                    ec=decade_colors.get(next_decade, "gray")
                )
            
            # plot anchor words in centroid plot too
            if anchor_embeddings:
                anchor_words = list(anchor_embeddings.keys())
                colors = [cmap(i/len(anchor_words)) for i in range(len(anchor_words))]
                
                for i, word in enumerate(anchor_words):
                    plt.scatter(
                        vis_anchors[i, 0],
                        vis_anchors[i, 1],
                        label=f"Anchor: {word}",
                        color=colors[i],
                        marker='*',
                        s=150,
                        edgecolors='black'
                    )
                    
                    # add text labels to anchor words
                    plt.annotate(
                        word,
                        (vis_anchors[i, 0], vis_anchors[i, 1]),
                        fontsize=9,
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
            
            plt.title(f"semantic shift of '{self._keyword}' centroids over decades (pca)")
            plt.xlabel(f"pc1 ({variance_explained[0]:.1%} variance)")
            plt.ylabel(f"pc2 ({variance_explained[1]:.1%} variance)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle="--", alpha=0.7)
            
            # save the centroid plot
            plt.savefig(f"plots/{self._keyword}_semantic_centroids_pca.pdf", dpi=300, bbox_inches="tight")
            logging.info(f"centroid plot saved to plots/{self._keyword}_semantic_centroids_pca.pdf")

        # if we have enough data, also create a 3D plot with PCA
        if len(all_embeddings) >= 5:
            try:
                # 3D PCA
                from mpl_toolkits.mplot3d import Axes3D
                
                pca3d = PCA(n_components=3)
                combined_embeddings_3d = np.vstack([all_embeddings, [emb for emb in anchor_embeddings.values()]]) if anchor_embeddings else all_embeddings
                
                vis_combined_3d = pca3d.fit_transform(combined_embeddings_3d)
                vis_embeddings_3d = vis_combined_3d[:len(all_embeddings)]
                if anchor_embeddings:
                    vis_anchors_3d = vis_combined_3d[len(all_embeddings):]
                    
                variance_explained_3d = pca3d.explained_variance_ratio_
                
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                for decade in sorted(set(embedding_decades)):
                    decade_indices = [i for i, d in enumerate(embedding_decades) if d == decade]
                    if decade_indices:
                        ax.scatter(
                            vis_embeddings_3d[decade_indices, 0],
                            vis_embeddings_3d[decade_indices, 1],
                            vis_embeddings_3d[decade_indices, 2],
                            label=f"{decade} (n={len(decade_indices)})",
                            color=decade_colors.get(decade, "gray"),
                            alpha=0.7
                        )
                
                # plot anchor words in 3D
                if anchor_embeddings:
                    anchor_words = list(anchor_embeddings.keys())
                    colors = [cmap(i/len(anchor_words)) for i in range(len(anchor_words))]
                    
                    for i, word in enumerate(anchor_words):
                        ax.scatter(
                            vis_anchors_3d[i, 0],
                            vis_anchors_3d[i, 1],
                            vis_anchors_3d[i, 2],
                            label=f"Anchor: {word}",
                            color=colors[i],
                            marker='*',
                            s=150,
                            edgecolors='black'
                        )
                        
                        # add text labels in 3D
                        ax.text(
                            vis_anchors_3d[i, 0],
                            vis_anchors_3d[i, 1],
                            vis_anchors_3d[i, 2],
                            word,
                            fontsize=9
                        )
                
                ax.set_title(f"3d semantic trajectory of '{self._keyword}' (pca)")
                ax.set_xlabel(f"pc1 ({variance_explained_3d[0]:.1%})")
                ax.set_ylabel(f"pc2 ({variance_explained_3d[1]:.1%})")
                ax.set_zlabel(f"pc3 ({variance_explained_3d[2]:.1%})")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.savefig(f"plots/{self._keyword}_semantic_trajectory_pca_3d.pdf", dpi=300, bbox_inches="tight")
                logging.info(f"3d plot saved to plots/{self._keyword}_semantic_trajectory_pca_3d.pdf")
            except Exception as e:
                logging.warning(f"failed to create 3d plot: {e}")
