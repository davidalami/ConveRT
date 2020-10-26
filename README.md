# ConveRT dual encoder model
Usage example:
```
import tensorflow_hub as tfhub
import tensorflow as tf

class ContextEncoder:
    def __init__(self):

        self.sess = tf.Session()
        self.module = tfhub.Module("https://github.com/davidalami/ConveRT/tf_model/model.tar.gz")

        self.text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
        self.extra_text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])

        self.context_encoding_tensor = self.module(
                                                    {
                                                       'context': self.text_placeholder,
                                                       'extra_context': self.extra_text_placeholder,
                                                    },
                                                        signature="encode_context"
                                                  )

        self.sess.run(tf.tables_initializer())
        self.sess.run(tf.global_variables_initializer())


    def encode_context(self, dialogue_history):

        context = dialogue_history[-1]

        extra_context = list(dialogue_history[:-1])
        extra_context.reverse()
        extra_context_feature = " ".join(extra_context)

        return self.sess.run(
                             self.context_encoding_tensor,
                             feed_dict={
                             self.text_placeholder: [context],
                             self.extra_text_placeholder: [extra_context_feature],
                             }
                            )[0]


# initialize a new Context Encoder object
ce = ContextEncoder()

# take a list of phrases, comprising a dialogue
dialogue = ["Hello, how are you?", "Hello, I am fine, thanks", "Glad to hear that!"]

# encode the dialogue to a 512-dimension vector
encodings = ce.encode_context(dialogue)

```
