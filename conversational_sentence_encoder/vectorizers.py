import tensorflow_hub as tfhub
import tensorflow_text  # NOQA: it is used when importing the model
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from .definitions import nocontext_model
from .definitions import multicontext_model

# The following setting allows the TF1 model to run in TF2
tf.compat.v1.disable_eager_execution()

# setting the logging verbosity level to errors-only
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class SentenceEncoder:
    """A client for running inference with a ConveRT encoder model.

        This wraps tensorflow hub, and gives an interface to input text, and
        get numpy encoding vectors in return. It includes a few optimizations to
        make encoding faster: deduplication of inputs, caching, and internal
        batching.
    """

    def __init__(self,
                 multiple_contexts=True,
                 batch_size=32,
                 ):

        self.multiple_contexts = multiple_contexts
        self.batch_size = batch_size

        self.sess = tf.compat.v1.Session()

        self.text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])


        if self.multiple_contexts:
            self.module = tfhub.Module(multicontext_model)
            self.extra_text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
            self.context_encoding_tensor = self.module(
                {
                    'context': self.text_placeholder,
                    'extra_context': self.extra_text_placeholder,
                },
                signature="encode_context"
            )

        else:
            self.module = tfhub.Module(nocontext_model)
            self.context_encoding_tensor = self.module(self.text_placeholder, signature="encode_context")
            self.encoding_tensor = self.module(self.text_placeholder)

        self.response_encoding_tensor = self.module(self.text_placeholder, signature="encode_response")
        self.sess.run(tf.compat.v1.tables_initializer())
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def encode_multicontext(self, dialogue_history):
        """Encode the whole dialogue to the encoding space to 512-dimensional vectors"""
        if not self.multiple_contexts:
            raise NotImplementedError("Can't encode multiple contexts using a noncontext model")

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
        )

    def encode_sentences(self, sentences):
        """Encode the given texts to the encoding space to 1024-dimensional vectors"""
        return self.batch_process(lambda x: self.sess.run(
            self.encoding_tensor, feed_dict={self.text_placeholder: x}
        ), sentences)

    def encode_contexts(self, sentences):
        """Encode the given context texts to the encoding space to 512-dimensional vectors"""
        return self.batch_process(lambda x: self.sess.run(
            self.context_encoding_tensor, feed_dict={self.text_placeholder: x}
              ), sentences)

    def encode_responses(self, sentences):
        """Encode the given response texts to the encoding space to 512-dimensional vectors"""
        return self.batch_process(
            lambda x: self.sess.run(
                self.response_encoding_tensor, feed_dict={self.text_placeholder: x}
            ),
            sentences)

    def batch_process(self, func, sentences):
        encodings = []
        for i in tqdm(range(0, len(sentences), self.batch_size), "encoding sentence batches"):
            encodings.append(func(sentences[i:i + self.batch_size]))
        return SentenceEncoder.l2_normalize(np.vstack(encodings))

    @staticmethod
    def l2_normalize(encodings):
        """L2 normalizes the given matrix of encodings."""
        norms = np.linalg.norm(encodings, ord=2, axis=-1, keepdims=True)
        return encodings / norms
