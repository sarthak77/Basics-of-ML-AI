#import modules
import gensim.downloader as api

#load pre-trained word-vectors from gensim-data
word_vectors = api.load("glove-wiki-gigaword-100")

result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['paris', 'england'], negative=['france'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['bright', 'night'], negative=['day'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['winter', 'hot'], negative=['summer'])
print("{}: {:.4f}".format(*result[0]))