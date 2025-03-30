"""Get the top frequent keywords from the extracted keywords."""
import json
from collections import Counter


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

def common_keywords_across_decades(decade_counts):
    common_words = set(decade_counts[next(iter(decade_counts))].elements())
    for words in decade_counts.values():
        common_words.intersection_update(words.elements())
    common_count = Counter({word: sum(decade[word] for decade in decade_counts.values()) for word in common_words})
    return common_count.most_common(20)

def main():
    # Load the data
    keywords_data = load_json('data/keywords.json')
    topics_data = load_json('data/topics.json')

    # Process the data
    keywords_decade_counts = process_keywords(keywords_data)
    topics_decade_counts = process_topics(topics_data)

    # Find common words across both datasets by decade
    common_keywords = common_keywords_across_decades(keywords_decade_counts)
    print("Common keywords across all decades in Keywords:")
    print(common_keywords)

if __name__ == "__main__":
    main()
