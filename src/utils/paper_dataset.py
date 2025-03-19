"""Paper dataset class for loading paper data, and related ."""
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List

import bertopic
import datasets
import nltk


class PaperDataset:
    def __init__(self, keyword: str) -> None:
        """Initialize the PaperDataset class."""
        # Dict[str, datasets.arrow_dataset.Dataset]
        self._ds = datasets.load_dataset("Seed42Lab/AI-paper-crawl")
        self._keyword = keyword

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
        return self._ds[conference_name]
    
    def filter_papers(self,
                      conference_name: str = None,
                      in_full_text: bool = True,
                      get_section: str = 'full_text'
                      ) -> Dict[str, List[datasets.arrow_dataset.Dataset]]:
        """Filter the papers based on the keyword.

        Args:
            conference_name     : conference to search for.
            in_full_text        : if the keyword must be in the full text or only in the abstract.
            get_section         : the section to get, 'abstract', 'full_text', 'main_text', 'reference'.

        Returns:
            papers              : the papers retrieved.
        """
        # configs
        if conference_name is None:
            conference_name = self.conference_list

        # get papers
        logging.info(f"Filtering the papers with the keyword '{self._keyword}'.")
        papers = defaultdict(list)
        for conference in conference_name:
            for paper in self._ds[conference]:
                text = self._segment_text(paper['text'], 'full_text' if in_full_text else 'abstract')
                if self._keyword in text:
                    papers[conference].append(paper)

        # filter sections
        logging.info(f"Segmenting the paper text into {get_section}.")
        papers_filtered = defaultdict(list)
        for conference in conference_name:
            for paper in papers[conference]:
                paper_tmp = paper.copy()
                text = self._segment_text(paper['text'], get_section)
                paper_tmp['text'] = text
                papers_filtered[conference].append(paper_tmp)

        return papers_filtered

    def _segment_text(self, 
                      text: str, 
                      get_section: str
                      ) -> Dict[str, List[str]]:
        """Segment the text into sentences.

        Args:
            text                : the text to segment.
            get_section         : the section to get, 'abstract', 'full_text', 'main_text', 'references'.

        Returns:
            papers              : the papers retrieved.
        """
        if get_section == 'abstract':
            # find beginning
            begin_idx = text.lower().find('abstract')
            if begin_idx == -1:
                return ''
            # find ending
            text_tmp = text[begin_idx:]
            while 'keywords' in text_tmp.lower():
                text_tmp = text_tmp[:text_tmp.lower().find('keywords')]
            while 'introduction' in text_tmp.lower():
                text_tmp = text_tmp[:text_tmp.lower().find('introduction')]
            return text_tmp
        elif get_section == 'full_text':
            return text
        elif get_section == 'main_text':
            # find beginning         
            begin_idx = text.lower().find('introduction')
            if begin_idx == -1:
                begin_idx = 0
            # find ending
            end_idx = text.lower().find('reference')
            if end_idx == -1:
                end_idx = len(text)
            return text[begin_idx:end_idx]
        elif get_section == 'reference':
            # find beginning
            text_tmp = text
            while 'references' in text_tmp.lower():
                text_tmp = text_tmp[text_tmp.lower().find('reference')+len('references'):]
            while 'reference' in text_tmp.lower():
                text_tmp = text_tmp[text_tmp.lower().find('reference')+len('reference'):]
            return text_tmp
        else:
            raise ValueError("Invalid section.")

    def get_topics(self,
                   conference_name: list = None,
                   in_full_text: bool = True,
                   min_topic_size: int = 10,
                   nr_topics: int = 'auto'
                   ) -> Dict[str, Any]:
        """Get the topics in the papers with BERTopic.

        Args:
            conference_name     : conference to search for, default all.
            in_full_text        : if the keyword must be in the full text or only in the abstract.
            min_topic_size      : minimum size of the topic.
            nr_topics           : the number of topics to get.

        Returns:
            topics              : the topics in the papers.
        """
        # configs
        if conference_name is None:
            conference_name = self.conference_list

        # build topic model
        logging.info("Building the topic model.")
        topic_model = bertopic.BERTopic(min_topic_size=min_topic_size, nr_topics=nr_topics)

        # get papers
        papers = self.get_paper_full_text(conference_name, in_full_text)

        # main loop
        topics = defaultdict(list)
        logging.info("Extracting the topics.")
        topics = defaultdict(list)
        for conf in papers:
            topics[conf] = []
            for paper in papers[conf]:
                topics, _ = topic_model.fit_transform(paper['text'])
                topic_info = topic_model.get_topic_info()
                topic_reprensentation = {}
                for topic_id in set(topics):
                    if topic_id != -1:
                        topic_words = topic_model.get_topic(topic_id)
                        topic_reprensentation[topic_id] = topic_words
                topics[conf].append({
                    'paper_id': paper['index'],
                    'topic_info': topic_info,
                    'topic_reprensentation': topic_reprensentation
                    'topics': topics
                    })
        return topics

    def get_related_words(self,
                          conference_name: list = None,
                          in_full_text: bool = True,
                          method: str = 'llm',
                          ) -> Dict[str, Dict[str, float]]:
        """Get the related words for the given keyword.

        Args:
            conference_name     : conference to search for, default all.
            in_full_text        : if the keyword must be in the full text or only the abstract.
            method      : the method to use, 'llm', 'sliding_window', 'parsing'.

        Returns:
            related_words       : the related words, in conf: keyword: freq.
        """
        # configs
        if conference_name is None:
            conference_name = self.conference_list

        pass

    def get_percent_numbers(self,
                            conference_name: list = None,
                            in_full_text: bool = True,
                            ) -> Dict[str, float]:
        """Get the percentage of numbers in selected papers.

        Args:
            conference_name     : conference to search for, default all.
            in_full_text        : if the keyword must be in the full text.

        Returns:
            percent_numbers     : the percentage of numbers.
        """
        # configs
        if conference_name is None:
            conference_name = self.conference_list

        # TODO: exclude the functional numbers, i.e., the numbers in the author list and references
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Main module of the application.')
    parser.add_argument('--keyword', '-k', type=str, help='The keyword to search for.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    paper_dataset = PaperDataset(keyword=args.keyword)

    papers = paper_dataset.filter_papers(conference_name=['ACL'], in_full_text=True, get_section='reference')

