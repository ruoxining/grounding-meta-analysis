import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# create output directory if it doesn't exist
if not os.path.exists('wordcloud_output'):
    os.makedirs('wordcloud_output')

# load the topics data
with open('data/topics.json', 'r') as f:
    data = json.load(f)

# prepare data structure for decades and overall
decades = {
    '1980s': defaultdict(float),
    '1990s': defaultdict(float),
    '2000s': defaultdict(float),
    '2010s': defaultdict(float),
    '2020s': defaultdict(float)
}
overall_topics = defaultdict(float)

# process each year's topic representation
for year_str, year_data in data.items():
    year = int(year_str.split('_')[0])
    
    # determine decade
    if 1980 <= year < 1990:
        decade = '1980s'
    elif 1990 <= year < 2000:
        decade = '1990s'
    elif 2000 <= year < 2010:
        decade = '2000s'
    elif 2010 <= year < 2020:
        decade = '2010s'
    elif 2020 <= year < 2030:
        decade = '2020s'
    else:
        continue  # skip if outside our decade range
    
    # process each topic
    if 'topic_reprensentation' in year_data:
        for topic_id, word_weights in year_data['topic_reprensentation'].items():
            for word, weight in word_weights:
                # add to decade-specific dictionary
                decades[decade][word] += weight
                # add to overall dictionary
                overall_topics[word] += weight

# create overall wordcloud
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white',
                      max_words=100,
                      collocations=False).generate_from_frequencies(overall_topics)

# plot and save overall wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Overall Topics Wordcloud')
plt.tight_layout()
plt.savefig('assets/overall_topics_wordcloud.pdf', dpi=300)
plt.close()

# create word clouds for each decade
for decade, word_weights in decades.items():
    if not word_weights:  # skip if no data for this decade
        continue
    
    # create wordcloud
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          max_words=50,
                          collocations=False).generate_from_frequencies(word_weights)
    
    # plot and save
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topics Wordcloud - {decade}')
    plt.tight_layout()
    plt.savefig(f'assets/topics_wordcloud_{decade}.pdf', dpi=300)
    plt.close()

print("Topic wordclouds generated and saved to 'wordcloud_output' directory.")
