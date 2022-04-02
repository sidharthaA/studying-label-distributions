
import pandas as pd

import nltk
from nltk import *
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = stopwords.words("english")
nltk.download('wordnet')
tknzr = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

df = pd.read_excel(r'final_cluster.xlsx')

grouped_df = df.groupby("Clusters")
grouped_lists = grouped_df[1].apply(list)

clust_dict = {}

for i in range(len(grouped_lists)):
    tweetlist = ""
    for j in grouped_lists[i]:
        tweetlist += str(j)
    clust_dict[i] = tweetlist

print(clust_dict)

preprocessedList = {}
i = 0
for text in clust_dict.values():
    if len(text.strip()) > 0:
        finaltext = []
        finalWords = ""
        words = tknzr.tokenize(text)
        for word in words:
            if word not in stopwords and len(word) > 3:
                lem = lemmatizer.lemmatize(word)
                stem = ps.stem(lem)
                if stem not in stopwords and len(stem)> 3:
                    finaltext.append(stem)
        finalWords = ' '.join([word for word in finaltext])
        preprocessedList[i] = finaltext
        i+=1

print(preprocessedList)

for i in preprocessedList.values():
    print(pd.Series(i).value_counts().sort_values(ascending=False).head(10))
