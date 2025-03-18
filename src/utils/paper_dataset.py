"""Paper dataset class for loading paper data, and related ."""
import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import datasets
import nltk
import tqdm
from bertopic import BERTopic

from utils.querier import Querier


class PaperDataset:
    def __init__(self, keyword: str) -> None:
        """Initialize the PaperDataset class."""
        # Dict[str, datasets.arrow_dataset.Dataset]
        self._ds = datasets.load_dataset("Seed42Lab/AI-paper-crawl")
        self._querier = Querier(model_name='gpt-4', api_key='api_key') # TODO: to real model and api key
        self._keyword = keyword
        self._lemmatizer = nltk.stem.WordNetLemmatizer()
        self._stopwords = set(nltk.corpus.stopwords.words('english'))

    @property
    def conference_list(self) -> List[str]:
        """Return the list of conferences."""
        return self._ds.keys()

    @property
    def keyword(self) -> str:
        """Return the keyword."""
        return self._keyword
    
    def get_conference(self, conference_name: str) -> List[str]:
        """Return the dataset for the given conference."""
        # TODO: to see if correct
        return self._ds[conference_name]
    
    def get_paper_by_keyword(self,
                             conference_name: list = None,
                             section: str = 'abstract'
                             ) -> Dict[str, List[datasets.arrow_dataset.Dataset]]:
        """Return the retrieved papers by keyword.
        
        Args:
            keyword         : keyword to be searched for.
            conference_name : specify a list of conference names to search for, default all.
            section         : specify the section to search for, default 'abstract', choices: ['abstract', 'title', 'full_text'].

        Returns:
            papers          : a dictionary of papers retrieved by keyword.
        """
        # configs
        papers = defaultdict(list)
        if conference_name is None:
            conference_name = self.conference_list
        assert section in ['abstract', 'title', 'full_text'], "Invalid section."        

        def get_section(paper, section):
            if section == 'abstract':
                # TODO: some paper does not have abstract
                try:
                    return paper['text'].lower().split('introduction')[0].split('abstract')[1]
                except:
                    return ''
            elif section == 'title':
                return paper['text'].lower().split('abstract')[0]
            elif section == 'full_text':
                return paper['text'].lower()

        # main loop
        for conference in conference_name:
            for paper in self._ds[conference]:
                if self.keyword in get_section(paper, section):
                    papers[conference].append(paper)

        return papers

    def get_related_words(self, window_size: int, top_n: int) -> Dict[str, List[Tuple[str, int]]]:
        """Get the related words for the given keyword with LLMs.
        
        Args:
            window_size     : window size to search from.
            top_n           : number of top related words to return.

        Returns:
            related_words   : the list of related words.
        """
        logging.info('Getting related words...')
        paper_by_conference = self.get_paper_by_keyword(section='full_text')
        related_words = {}

        logging.info('Building word frequency...')
        for conference, papers in paper_by_conference.items():
            logging.info(f'Processing conference: {conference}')
            word_freq = defaultdict(int)

            for paper in tqdm.tqdm(papers):
                text = paper['text'].lower()
                sentences = nltk.tokenize.sent_tokenize(text)

                for sentence in sentences:
                    if self.keyword in sentence:
                        words = nltk.tokenize.word_tokenize(sentence)
                        keyword_indices = [i for i, word in enumerate(words) if word == self.keyword]

                        for idx in keyword_indices:
                            start = max(0, idx - window_size)
                            end = min(len(words), idx + window_size + 1)

                            # get related words in the window
                            for i in range(start, end):
                                if i != idx:  # skip the keyword itself
                                    word = words[i]
                                    word = re.sub(r'[^\w\s]', '', word)
                                    if word and word not in self._stopwords:
                                        word = self._lemmatizer.lemmatize(word)
                                        word_freq[word] += 1

            # TODO: better logging everywhere

            # get top n words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
            related_words[conference] = top_words
    
        # Conf: List[(word, freq)]

        return related_words

    def get_related_topics(self, 
                           min_topic_size: int = 10, 
                           nr_topics: str = 'auto'
                           ) -> List[str]:
        """Get the related topics for the given keyword with BERTopic.

        Args:
            min_topic_size  : minimum size of topics.
            nr_topics       : number of topics to extract, 'auto' for automatic detection.

        Returns:
            related_topics  : the list of related topics.
        """
        papers_by_conf = self.get_paper_by_keyword(section='abstract')
        results = {}

        for conf, papers in papers_by_conf.items():
            # get abstracts
            abstracts = []
            for paper in papers:
                text = paper['text'].lower()
                if 'abstract' in text and 'introduction' in text:
                    abstract = text.split('introduction')[0].split('abstract')[1].strip()
                    abstracts.append(abstract)

            if len(abstracts) < min_topic_size:
                results[conf] = {"status": "Not enough papers."}

            # create and fit BERTopic model
            topic_model = BERTopic(min_topic_size=min_topic_size, nr_topics=nr_topics)
            topics, _ = topic_model.fit_transform(abstracts)

            # get topic information
            topic_info = topic_model.get_topic_info()
            topic_representations = {}

            for topic_id in set(topics):
                # skip        
                if topic_id != -1:
                    topic_words = topic_model.get_topic(topic_id)
                    topic_representations[topic_id] = topic_words

            results[conf] = {
                "status": "success",
                "topic_info": topic_info,
                "topic_representations": topic_representations,
                "topics": topics,
                "model": topic_model
            }

        return results

    def extract_related_papers(self) -> Dict[str, List[Dict[str, str]]]:
        """Extract the related papers for the given keyword.
        
        Returns:
            related_papers  : the related papers, both references and derivations.
        """
        pass
