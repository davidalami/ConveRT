# Conversational Sentence Encoder

This project features the ConveRT dual-encoder model, using subword representations 
and lighter-weight more efficient transformer-style blocks to encode text, 
as described in the [ConveRT paper](https://arxiv.org/abs/1911.03688). 
It provides powerful representations for conversational data, 
and can also be used as a response ranker. 
Also it features the multi-context ConveRT model, that uses extra contexts 
from the conversational history to refine the context representations. 
The extra contexts are the previous messages in the dialogue 
(typically at most 10) prior to the immediate context.

# Installation 
Just pip install the package and you are ready to go!
```
pip install conversational-sentence-encoder
```

# Usage examples
The entry point of the package is SentenceEncoder class:
```
from conversational_sentence_encoder.vectorizers import SentenceEncoder
```
To run the examples you will also need to 
```
pip install scikit-learn
```
## Text Classification / Intent Recognition / Sentiment Classification
The ConveRT model encodes sentences to a meaningful semantic space. Sentences can be compared for semantic similarity in this space, and NLP classifiers can be trained on top of these encodings
```
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# texts
X = ["hello ? can i speak to a actual person ?",
"i ll wait for a human to talk to me",
"i d prefer to be speaking with a human .",
"ok i m not talking to a real person so",
"ok . this is an automated message . i need to speak with a real human",
"hello . i m so sorry but i d like to return . can you send me instructions . thank you",
"im sorry i m not sure you understand what i need but i need to return a package i got from you guys",
"how can i return my order ? no one has gotten back to me on here or emails i sent !",
"i can t wait that long . even if the order arrives i will send it back !",
"i have a question what is your return policy ? i ordered the wrong size"]

# labels
y = ["SPEAK_HUMAN"]*5+["RETURN"]*5

# initialize the ConveRT dual-encoder model
sentence_encoder = SentenceEncoder(multiple_contexts=False)

# output 1024 dimensional vectors, giving a representation for each sentence. 
X_encoded = sentence_encoder.encode_sentences(X)

# encode labels
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y)

# fit the KNN classifier on the toy dataset
clf = KNeighborsClassifier(n_neighbors=3).fit(X_encoded, y_encoded)


test = sentence_encoder.encode_sentences(["are you all bots???", 
                                          "i will send this trash back!"])

prediction = clf.predict(test)

# this will give the intents ['SPEAK_HUMAN' 'RETURN']
print(le.inverse_transform(prediction))
```
## Response Selection (Neural Ranking)
```
# outputs 512 dimensional vectors, giving the context representation of each input. 
#These are trained to have a high cosine-similarity with the response representations of good responses
sentence_encoder.encode_contexts(dialogue)

# outputs 512 dimensional vectors, giving the response representation of each input. 
#These are trained to have a high cosine-similarity with the context representations of good corresponding contexts
sentence_encoder.encode_responses(dialogue)


# initialize the multi-context ConveRT model, that uses extra contexts from the conversational history to refine the context representations
multicontext_encoder = SentenceEncoder(multiple_contexts=True)

# outputs 512 dimensional vectors, giving the whole dialogue representation
multicontext_encoder.encode_multicontext(dialogue)
```
# Notes
This project is a continuation of the abandoned https://github.com/PolyAI-LDN/polyai-models,
it is distributed under the same license. 