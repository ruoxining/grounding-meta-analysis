"""Paper dataset class for loading paper data, and related ."""
from collections import defaultdict
from typing import Dict, List

import datasets

from utils.querier import Querier


class PaperDataset:
    def __init__(self, keyword: str) -> None:
        """Initialize the PaperDataset class."""
        # Dict[str, datasets.arrow_dataset.Dataset]
        self._ds = datasets.load_dataset("Seed42Lab/AI-paper-crawl")
        self._querier = Querier(model_name='gpt-4', api_key='api_key') # TODO: to real model and api key
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
                return paper['text'].lower().split('introduction')[0].split('abstract')[1]
            elif section == 'title':
                return paper['text'].lower().split('abstract')[0]
            elif section == 'full_text':
                return paper['text'].lower

        # main loop
        for conference in conference_name:
            for paper in self._ds[conference]:
                if self.keyword in get_section(paper, section):
                    papers[conference].append(paper)

        return papers

    def get_related_words(self, window_size: int) -> List[str]:
        """Get the related words for the given keyword with LLMs.
        
        Args:
            window_size     : the window size to search from.

        Returns:
            related_words   : the list of related words.
        """
        pass

    def get_related_topics(self) -> List[str]:
        """Get the related topics for the given keyword with BERTopic.

        Returns:
            related_topics  : the list of related topics.
        """
        pass

    def model_word_clusters(self,
                            if_center: bool 
                            ) -> None:
        """Model the word clusters and visualize them.
        
        Args:
            if_center       : if the clusters should be modeled with the keyword or not.
        """
        pass

    def extract_related_papers(self) -> Dict[str, List[Dict[str, str]]]:
        """Extract the related papers for the given keyword.
        
        Returns:
            related_papers  : the related papers, both references and derivations.
        """
        pass

    def model_paper_graph(self) -> None:
        """Model the paper graph and visualize it."""
        pass

    def model_trend_graph(self) -> None:
        """Model the trend graph and visualize it."""
        pass
