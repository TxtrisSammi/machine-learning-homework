from urllib.request import urlopen as get
from collections import Counter
import re

url = 'https://www.gutenberg.org/files/20727/20727.txt'

with get(url) as response:
    # Using re.findall is much faster than iterating over every character
    text = response.read().decode('utf-8').lower()
    words = re.findall(r'[a-z]{7,}', text) # Finds words with 7+ letters directly

# Counter does the heavy lifting for you
word_counts = Counter(words)

for word, count in word_counts.most_common(10):
    print(f"{word}: {count}")
