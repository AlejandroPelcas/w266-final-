# Author: Laura Wendlandt
# Dependencies: nltk, gensim
# Example: Calculating stability for a word across two word2vec spaces

from stability import mostSimilar,stability
from nltk.corpus import brown
from gensim.models import word2vec

sentences = brown.sents()
sentences = [[word.lower() for word in sentence] for sentence in sentences]

print("Training two word2vec models on the Brown corpus")
model1 = word2vec.Word2Vec(sentences,vector_size=300,window=5,min_count=1,seed=42)
model2 = word2vec.Word2Vec(sentences,vector_size=300,window=5,min_count=1,seed=102)

print("Formatting models for stability code")
all_words = set([word for sentence in sentences for word in sentence])
model1_dict = {}
model2_dict = {}
for word in all_words:
	model1_dict[word] = model1.wv[word]
	model2_dict[word] = model2.wv[word]

print('Calculating stability for the word "woman"')
mostSimilar1 = mostSimilar(model1_dict,'woman')
mostSimilar2 = mostSimilar(model2_dict,'woman')

print('Calculating stability')
stab = stability('woman',[mostSimilar1,mostSimilar2],[mostSimilar1,mostSimilar2],True)
print('"woman" has a stability of ' + str(stab*10) + '%')
