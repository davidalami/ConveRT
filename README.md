# Conversational Sentence Encoder 
[![GitHub license](https://img.shields.io/github/license/davidalami/ConveRT)](https://github.com/davidalami/ConveRT/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/davidalami/ConveRT)](https://github.com/davidalami/ConveRT/issues)
[![GitHub forks](https://img.shields.io/github/forks/davidalami/ConveRT)](https://github.com/davidalami/ConveRT/network)
[![GitHub stars](https://img.shields.io/github/stars/davidalami/ConveRT)](https://github.com/davidalami/ConveRT/stargazers)

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
Just pip install the package (works for python 3.6.* and 3.7.*) and you are ready to go!
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
## Text Classification / Intent Recognition / Sentiment Classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidalami/ConveRT/blob/main/examples/text_classification.ipynb)
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
## Response Selection (Neural Ranking) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidalami/ConveRT/blob/main/examples/response_selection.ipynb)
ConveRT is trained on the response ranking task, so it can be used to find good responses to a given conversational context.

This section demonstrates how to rank responses, by computing cosine similarities of context and response representations in the shared response ranking space. Response representations for a fixed candidate list are first pre-computed. When a new context is provided, it is encoded and then compared to the pre-computed response representations.
```
import numpy as np

# initialize the ConveRT dual-encoder model
sentence_encoder = SentenceEncoder(multiple_contexts=False)

questions = np.array(["where is my order?", 
                      "what is the population of London?",
                      "will you pay me for collaboration?"])

# outputs 512 dimensional vectors, giving the context representation of each input. 
#These are trained to have a high cosine-similarity with the response representations of good responses
questions_encoded = sentence_encoder.encode_contexts(questions)


responses = np.array(["we expect you to work for free",
                      "there are a lot of people",
                      "its on your way"])

# outputs 512 dimensional vectors, giving the response representation of each input. 
#These are trained to have a high cosine-similarity with the context representations of good corresponding contexts
responses_encoded = sentence_encoder.encode_responses(responses)

# computing pairwise similarities as a dot product
similarity_matrix = questions_encoded.dot(responses_encoded.T)

# indices of best answers to given questions
best_idx = np.argmax(similarity_matrix, axis=1)

# will output answers in the right order
# ['its on your way', 'there are a lot of people', 'we expect you to work for free']
print(np.array(responses)[best_idx])
```
## Multi-context response ranking/similarity/classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidalami/ConveRT/blob/main/examples/multi_context.ipynb)
This model takes extra dialogue history into account allowing to create smart conversational agents
```
import numpy as np
from conversational_sentence_encoder.vectorizers import SentenceEncoder

# initialize the multi-context ConveRT model, that uses extra contexts from the conversational history to refine the context representations
multicontext_encoder = SentenceEncoder(multiple_contexts=True)

dialogue = np.array(["hello", "hey", "how are you?"])

responses = np.array(["where do you live?", "i am fine. you?", "i am glad to see you!"])

# outputs 512 dimensional vectors, giving the whole dialogue representation
dialogue_encoded = multicontext_encoder.encode_multicontext(dialogue)

# encode response candidates using the same model
responses_encoded = multicontext_encoder.encode_responses(responses)

# get the degree of response fit to the existing dialogue
similarities = dialogue_encoded.dot(responses_encoded.T)

# find the best response
best_idx = np.argmax(similarities)

# will output "i am fine. you?"
print(responses[best_idx])
```
# Notes
This project is a continuation of the abandoned https://github.com/PolyAI-LDN/polyai-models,
it is distributed under the same license. 
