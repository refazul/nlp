import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# basic tokenize
text = "Natural language processing (NLP) is a field of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human (natural) languages."
tokens = word_tokenize(text)
fdist = FreqDist(tokens)
print(fdist.most_common(10))

# pos analysis
text = "Natural language processing is a fascinating area of study."
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)

# sentiment analysis
text = "I love natural language processing. It's incredibly interesting!"
sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(text)
print(sentiment)

# named entitiy recognition
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
text = "Google was founded by Larry Page and Sergey Brin while they were students at Stanford University. They started the company in September 1998 in a friend's garage in California."
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
#displacy.serve(doc, style="ent", port=5001)

# syntactic parsing
nlp = spacy.load('en_core_web_sm')
text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

for token in doc:
    print(f"{token.text:{15}} {token.pos_:{10}} {token.dep_:{10}} {token.head.text}")
displacy.serve(doc, style="dep", auto_select_port=True)

