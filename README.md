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
```
from conversational_sentence_encoder.vectorizers import SentenceEncoder


# initialize the ConveRT dual-encoder model
sentence_encoder = SentenceEncoder(multiple_contexts=False)

dialogue = ["Hello, how are you?", "Hello, I am fine, thanks", "Glad to hear that!"]

# outputs 1024 dimensional vectors, giving a representation for each sentence. 
sentence_encoder.encode_sentences(dialogue)

# outputs 512 dimensional vectors, giving the context representation of each input. These are trained to have a high cosine-similarity with the response representations of good responses
sentence_encoder.encode_contexts(dialogue)

# outputs 512 dimensional vectors, giving the response representation of each input. These are trained to have a high cosine-similarity with the context representations of good corresponding contexts
sentence_encoder.encode_responses(dialogue)


# initialize the multi-context ConveRT model, that uses extra contexts from the conversational history to refine the context representations
multicontext_encoder = SentenceEncoder(multiple_contexts=True)

# outputs 512 dimensional vectors, giving the whole dialogue representation
multicontext_encoder.encode_multicontext(dialogue)
```
# Notes
This project is a continuation of the abandoned https://github.com/PolyAI-LDN/polyai-models,
it is distributed under the same licence. 