import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# create output directory if it doesn't exist
if not os.path.exists('wordcloud_output'):
    os.makedirs('wordcloud_output')

# load the json data
with open('data/keywords.json', 'r') as f:
    data = json.load(f)

# prepare data structure for decades
decades = {
    '1980s': [],
    '1990s': [],
    '2000s': [],
    '2010s': [],
    '2020s': []
}

# prepare all_keywords for overall wordcloud
all_keywords = []

# process data
for conference, papers in data.items():
    for paper_id, paper_info in papers.items():
        year = int(paper_info['year'].split('_')[0])
        keywords = paper_info['related_words']
        
        # add to appropriate decade
        if 1980 <= year < 1990:
            decades['1980s'].extend(keywords)
        elif 1990 <= year < 2000:
            decades['1990s'].extend(keywords)
        elif 2000 <= year < 2010:
            decades['2000s'].extend(keywords)
        elif 2010 <= year < 2020:
            decades['2010s'].extend(keywords)
        elif 2020 <= year < 2030:
            decades['2020s'].extend(keywords)
        
        # add to overall list
        all_keywords.extend(keywords)

# create word frequency dictionary for overall
word_freq = defaultdict(int)
for word in all_keywords:
    if word != 'grounding' and word.isalpha():
        word_freq[word] += 1

# create overall wordcloud
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white',
                      max_words=100,
                      collocations=False).generate_from_frequencies(word_freq)

# plot and save overall wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Overall Keywords Wordcloud')
plt.tight_layout()
plt.savefig('assets/overall_keywords_wordcloud.pdf', dpi=300)
plt.close()

# create word clouds for each decade
for decade, keywords in decades.items():
    if not keywords:  # skip if no keywords for this decade
        continue
    
    # create word frequency dictionary
    decade_freq = defaultdict(int)
    for word in keywords:
        if word != 'grounding' and word.isalpha():
            decade_freq[word] += 1
    
    # create wordcloud
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          max_words=50,
                          collocations=False).generate_from_frequencies(decade_freq)
    
    # plot and save
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Keywords Wordcloud - {decade}')
    plt.tight_layout()
    plt.savefig(f'assets/keywords_wordcloud_{decade}.pdf', dpi=300)
    plt.close()

print("Keyword wordclouds generated and saved to 'wordcloud_output' directory.")
