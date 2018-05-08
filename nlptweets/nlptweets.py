import re, time, nltk
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from progress.bar import Bar
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

# TO DO:
# 1. Clean data more for NLP - stemming, lowercase, word count, stop words, etc.
# 2. Faster fuzzy matching

# Customize
file = 'skatekitchen'							# File Name
stopwrd = []


# Input Workbook 0.175184965133667
path = 'inputs/{}.csv'.format(file)
df = pd.read_csv(path)
# xl = pd.ExcelFile(path)
# df = xl.parse(0)

# Pre-Process Data
def processexcel(dataframe):
	dataframe['matches'] = False
	print('{} total rows.'.format(dataframe.shape[0]))
	try:
		dataframe = dataframe[dataframe.Type == 'Regular']
	except:
		dataframe = dataframe
	dataframe = dataframe[['Text', 'Tweet_URL', 'matches']]
	dataframe = dataframe[~dataframe['Text'].str.startswith(('RT',' RT','@',))]
	dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
	print('{} "regular" rows.'.format(dataframe.shape[0]))
	return dataframe

# Resort Index by A-Z
def resort(dataframe, value):
	dataframe = dataframe.sort_values(value)
	dataframe = dataframe.reset_index(drop=True)
	return dataframe

def randomize(dataframe):
	dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
	return dataframe

# START PROCESSING
df = processexcel(df)

# CLEAN DATA
httpcd = []
for row in df['Text']:
	httpc = re.sub(r'http\S+', '', row)
	httpc = httpc.lower()
	httpc = re.sub(' +', ' ', httpc)
	httpc = re.sub(r'#\S+','', httpc)
	httpc = re.sub(r'via @\S+','', httpc)
	httpc = re.sub(r'from @\S+','', httpc)
	httpc = re.sub(r'on @\S+','', httpc)
	httpc = re.sub(r'@\S+','', httpc)
	httpc = re.sub(r'I liked a @YouTube video ','', httpc)
	httpc = re.sub('[\W_]+', ' ', httpc)
	httpc = httpc.lstrip(' ')
	httpcd.append(httpc)
df['cleaned'] = httpcd


# FUZZY MATCH
total = df.shape[0]
fb = Bar('Processing', max=total, suffix='%(percent)d%% - %(eta)ds')
df = resort(df, 'Text')
df['cleaned_s'] = df['cleaned'].shift(-1)
df['cleaned_s'] = df['cleaned_s'].fillna(df.at[0,'cleaned_s'])
for i in df.index:
	main = df.at[i,'cleaned']
	sub = df.at[i,'cleaned_s']
	fuzzr = fuzz.ratio(main,sub)
	if fuzzr > 94:
		df.at[i,'matches'] = True
	fb.next()
before = df.shape[0]
df = df[df.matches != True]
df = resort(df, 'Text')
print('\n')

# NLTK
stemmer = SnowballStemmer('english')
stop = set(stopwords.words('english'))
stop.update(stopwrd)
df['stemmed'] = df['cleaned'].apply(word_tokenize)
df['stemmed'] = df['stemmed'].apply(lambda x: [item for item in x if item not in stop])
df['stemmed'] = df['stemmed'].apply(lambda x: [stemmer.stem(y) for y in x])

# Most Common
i = 0
count_all = Counter()
for row in df['stemmed']:
	terms_all = [term for term in df.at[i,'stemmed']]
	count_all.update(terms_all)
	i += 1
print(count_all.most_common(50))


# Export out unique tweets
print('\n')
print('{} total rows before fuzzy matching.'.format(before))
df = df[['Text', 'Tweet_URL', 'stemmed']]
df = randomize(df)
print('{} total rows after fuzzy matching.'.format(df.shape[0]))
writer = pd.ExcelWriter('outputs/{}_Output.xls'.format(file))
df.to_excel(writer, 'Unique', index = False)
writer.save()

