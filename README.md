# grounding-meta-analysis
Course project for UWaterloo CS784: Computational Linguistics.

python version: 3.10

How to run the code:

```bash
pip install -r requirements.txt

python src/main.py -k grounding -a key -e <experiment-name>
```

Supported experiment names:

- `all`: run all the following experiments.
- `keyword_model`: keyword extraction from full text (?)
- `topic_model`: topic modeling from full text (?)
- `coocuring_keywords`: see if several specific keywords are coocurring with the topic keyword.
- `percent_numbers`: model the percentage of numbers in the paper.
- `complexity_scores`: model the complexity score of the paper's writing style.
- `trend`: model the trend by year / by conference of the number of the papers.


# meanings of grounding

1. ontology: the relationship between a word and the world. (meaning and the world, symbol and the meaning)
2. symbol grounding problem: in artificial intelligence, cognitive science, philosophy of mind, and semantics. It addresses the challenge of connecting symbols, such as words or abstract representations to the real-world objects or concepts that they refer to.


# research methods involved

1. keywords extraction: extract keywords within a context-window of the keyword 'grounding' from the abstracts / methodology.
2. topic modeling: also extract topics from the abstracts with the keyword.
3. citation analysis: build a citation graph.
4. conference specific trend: analyze the trend of the grounding research in different conferences. (that flow graph)

