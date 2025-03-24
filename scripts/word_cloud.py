import json
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def process_keywords(data):
    decade_counts = {}
    for papers in data.values():
        for paper_id, info in papers.items():
            year = int(info['year'].split('_')[0])
            decade = (year // 10) * 10
            if decade not in decade_counts:
                decade_counts[decade] = Counter()
            decade_counts[decade].update([word.lower() for word in info['related_words']])
    return decade_counts

def process_topics(data):
    decade_counts = {}
    for year, details in data.items():
        decade = (int(year.split('_')[0]) // 10) * 10
        if decade not in decade_counts:
            decade_counts[decade] = Counter()
        for topic in details['topic_info']:
            decade_counts[decade].update([word.lower() for word in topic['Representation']])
    return decade_counts

def merge_decade_counts(keyword_counts, topic_counts):
    for decade, counts in topic_counts.items():
        if decade in keyword_counts:
            keyword_counts[decade] += counts
        else:
            keyword_counts[decade] = counts
    return keyword_counts

def create_wordclouds(decade_counts):
    for decade, counts in sorted(decade_counts.items()):
        plt.figure(figsize=(10, 8))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(counts)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {decade}s')
        plt.axis('off')
        plt.savefig(f'assets/wordcloud_{decade}.png')

def main():
    keywords_data = load_json('data/keywords.json')
    topics_data = load_json('data/topics.json')
    
    keywords_decade_counts = process_keywords(keywords_data)
    topics_decade_counts = process_topics(topics_data)
    
    merged_decade_counts = merge_decade_counts(keywords_decade_counts, topics_decade_counts)
    create_wordclouds(merged_decade_counts)

if __name__ == "__main__":
    main()
