from urllib.request import urlopen as get

url = 'https://www.gutenberg.org/files/20727/20727.txt'
with get(url) as response:
    text = response.read().decode('utf-8').lower()


filteredContents = ''
for char in text:
    if 'a' <= char and char <= 'z' or char == ' ':
        filteredContents += char
filteredContents = filteredContents.split(' ')

wordCount = {}
for word in filteredContents:
    if len(word) > 6:
        if word in wordCount:
            wordCount[word] += 1    
        else:
            wordCount[word] = 1
wordCount = sorted(wordCount.items(), key=lambda item: item[1], reverse=True)
    
for word, count in wordCount[:10]:
    print(f"{word}: {count}")
