"""Paper dataset class for loading paper data, and related ."""
import logging
from collections import defaultdict
from typing import Any, Dict, List

import bertopic
import datasets
import nltk
import textstat
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


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

    def group_papers(self,
                     by: str = 'year',
                     conference_name: str = None,
                     in_full_text: bool = True,
                     get_section: str = 'full_text'
                     ) -> Dict[str, List[str]]:
        """Group the filtered papers by the given domain.

        Args:
            by                  : the domain to group by, 'year' or 'conf'.
            conference_name     : conference to search for.
            in_full_text        : if the keyword must be in the full text or only in the abstract.
            get_section         : the section to get, 'abstract', 'full_text', 'main_text', 'reference'.

        Returns:
            paper_grouped       : the papers retrieved.
        """
        papers = self.filter_papers(conference_name=conference_name, in_full_text=in_full_text, get_section=get_section)

        logging.info(f"Grouping the papers by {by}.")
        paper_grouped = defaultdict(list)
        if by == 'year':
            for conf in papers:
                for paper in papers[conf]:
                    if len(paper['text'].strip()) == 0:
                        continue
                    paper_grouped[paper['year']].append(paper['text'])
        elif by == 'conf':
            for conf in papers:
                for paper in papers[conf]:
                    if len(paper['text'].strip()) == 0:
                        continue
                    paper_grouped[conf].append(paper['text'])
        else:
            raise ValueError("Invalid group by.")

        return paper_grouped

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
                   by: str = 'year',
                   conference_name: list = None,
                   in_full_text: bool = True,
                   min_topic_size: int = 2,
                   nr_topics: int = 'auto'
                   ) -> Dict[str, Dict[str, Any]]:
        """Get the topics in the papers with BERTopic.

        Args:
            by                  : the domain to group by, 'year' or 'conf'.
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
        stop_words = list(ENGLISH_STOP_WORDS)
        stop_words.extend(['et', 'al', 'fig', 'table', 'section', 'appendix', 'figure', 'reference', 'references', 'introduction', 'abstract', 'keywords', 'conclusion', 'acknowledgement', 'acknowledgements'])
        vectorizer = CountVectorizer(stop_words=stop_words)
        topic_model = bertopic.BERTopic(
            vectorizer_model=vectorizer,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics
            )

        # get papers
        papers = self.group_papers(
            by=by,
            conference_name=conference_name,
            in_full_text=in_full_text,
            get_section='full_text'
            )

        # main loop
        logging.info("Extracting the topics.")
        topics_all = defaultdict(dict)
        for year in papers:
            if len(papers[year]) <= min_topic_size:
                logging.debug('Skipping year %s due to lack of papers.', year)
                continue
            try:
                topics, _ = topic_model.fit_transform(papers[year])
            except:
                # TODO: why is this error
                logging.debug('Skipping year %s due to an error.???', year)
                continue
            topic_info = topic_model.get_topic_info()
            topic_reprensentation = {}
            for topic_id in set(topics):
                if topic_id != -1:
                    topic_words = topic_model.get_topic(topic_id)
                    topic_reprensentation[topic_id] = topic_words

            topics_all[year] = {
                'topic_info': topic_info,
                'topic_reprensentation': topic_reprensentation,
                'topics': topics
                }

        return topics_all

    def get_related_words(self,
                          conference_name: list = None,
                          in_full_text: bool = True,
                          method: str = 'llm'
                          ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Get the related words for the given keyword.

        Args:
            conference_name     : conference to search for, default all.
            in_full_text        : if the keyword must be in the full text or only the abstract.
            method      : the method to use, 'llm', 'cooccur'.

        Returns:
            related_words       : the related words, in conf: keyword: freq.
        """
        papers = self.filter_papers(conference_name=conference_name, in_full_text=in_full_text, get_section='full_text')

        logging.info("Extracting the related words.")
        related_words = defaultdict(dict)
        if method == 'llm':
            pass
        elif method == 'cooccur':
            for conf in papers:
                for paper in papers[conf]:
                    related_words[conf][paper['index']] = {
                        'year': paper['year'],
                        'related_words': []
                        }
                    sentences = nltk.sent_tokenize(paper['text'])
                    for sentence in sentences:
                        # get nouns from a sentence
                        words = nltk.word_tokenize(sentence)
                        pos_tags = nltk.pos_tag(words)
                        if self._keyword not in words:
                            continue
                        indices_noun = [i for i, (_, pos) in enumerate(pos_tags) if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
                        words_noun = [words[i] for i in indices_noun]
                        related_words[conf][paper['index']]['related_words'].extend(words_noun)

        return related_words

    def get_percent_numbers(self,
                            conference_name: list = None,
                            in_full_text: bool = True,
                            ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Get the percentage of numbers in selected papers.

        Args:
            conference_name     : conference to search for, default all.
            in_full_text        : if the keyword must be in the full text.

        Returns:
            percent_numbers     : the percentage of numbers.
        """
        papers = self.filter_papers(conference_name=conference_name, in_full_text=in_full_text, get_section='main_text')

        logging.info("Calculating the percentage of numbers.")
        percent_numbers = defaultdict(dict)
        for conf in papers:
            for paper in papers[conf]:
                if len(paper['text'].strip()) == 0:
                    continue
                total_words = len(paper['text'].split())
                total_numbers = sum(c.isdigit() for c in paper['text'])
                percent_numbers[conf][paper['index']] = {
                    'year': paper['year'],
                    'percent': total_numbers / total_words
                    }

        return percent_numbers

    def get_complexity_score(self,
                             conference_name: list = None,
                             in_full_text: bool = True,
                             method: str = 'fog'
                             ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Get the complexity score of the papers.

        Args:
            conference_name     : conference to search for, default all.
            in_full_text        : if the keyword must be in the full text.
            method              : method to use, 'fog', 'etymology'.

        Returns:
            complexity_scores   : complexity scores of each paper with years.
        """
        papers = self.filter_papers(conference_name=conference_name, in_full_text=in_full_text, get_section='full_text')

        logging.info(f"Calculating the complexity score with {method}.")
        complexity_scores = defaultdict(dict)
        if method == 'fog':
            for conf in papers:
                for paper in papers[conf]:
                    complexity_scores[conf][paper['index']] = {
                        'year': paper['year'],
                        'score': textstat.gunning_fog(paper['text'])
                        }
        elif method == 'etymology':
            pass

        return complexity_scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Main module of the application.')
    parser.add_argument('--keyword', '-k', type=str, help='The keyword to search for.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # TODO: to make no filtering, but will have some buggy behavior
    if args.keyword is None:
        args.keyword = ' '

    paper_dataset = PaperDataset(keyword=args.keyword)

    # topics = paper_dataset.get_topics(by='year', conference_name=['ACL'], in_full_text=True, min_topic_size=2, nr_topics='auto')

    # complexity_scores = paper_dataset.get_complexity_score(conference_name=['ACL'], in_full_text=True, method='fog')

    # percent_numbers = paper_dataset.get_percent_numbers(conference_name=['ACL'], in_full_text=True)

    related_words = paper_dataset.get_related_words(conference_name=['ACL'], in_full_text=True, method='cooccur')

    from IPython import embed; embed()
