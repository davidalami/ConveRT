# ConveRT dual encoder model
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
