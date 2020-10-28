# ConveRT dual-encoder model

This is the ConveRT dual-encoder model, using subword representations and lighter-weight more efficient transformer-style blocks to encode text, as described in the [ConveRT paper](https://arxiv.org/abs/1911.03688). It provides powerful representations for conversational data, and can also be used as a response ranker. 


# Multi-Context ConveRT
This is the multi-context ConveRT model from the [ConveRT paper](https://arxiv.org/abs/1911.03688), that uses extra contexts from the conversational history to refine the context representations. The extra contexts are the previous messages in the dialogue (typically at most 10) prior to the immediate context.


Usage example:
```
from context_encoder import ContextEncoder

# initialize a new Context Encoder object
ce = ContextEncoder()

# take a list of phrases, comprising a dialogue
dialogue = ["Hello, how are you?", "Hello, I am fine, thanks", "Glad to hear that!"]

# encode the dialogue to a 512-dimension vector
encodings = ce.encode_context(dialogue)

```
