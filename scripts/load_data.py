from datasets import load_dataset

ds = load_dataset("Seed42Lab/AI-paper-crawl") # Dict[str, datasets.arrow_dataset.Dataset]

n_papers_title = []
n_papers = []
n_papers_more = []

for conference_name in ds.keys():
    n_papers_title.append(len([paper for paper in ds[conference_name] if 'grounding' in paper['text'].lower().split('abstract')[0]]))

    n_papers.append(len([paper for paper in ds[conference_name] if 'grounding' in paper['text'].lower().split('introduction')[0]]))

    n_papers_more.append(len([paper for paper in ds[conference_name] if 'grounding' in paper['text'].lower()]))

print(ds.keys())
print(n_papers_title)
print([n_papers[i] - n_papers_title[i] for i in range(len(n_papers))])
print(n_papers_more)
