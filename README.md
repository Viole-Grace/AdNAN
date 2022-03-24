# AdNAN
Adaptive coNtextuAl taxoNomy - AdNAN

## Self Supervision:

SSL (Self Supervised Learning) enables models to learn from large amounts of data, using underlying structure of data patterns. In text, these data patterns help us understand the 'relationship' between words and different representation of words with respect to each other.

Deep Neural Networks 'encode' this information using large amounts of data, and perform exceedingly well for tasks that require domain-wise adaptation or 'fine-tuning' for downstream tasks. In NLP, self-supervision has led to the development of models such as Word2Vec, fastText, GloVe, BERT, etc. 

However, representation in text is also a function of how spoken or written language works -- specifically, an underlying grammatical structure to give 'meaning' to pieces of text. This structure can be understood (to some extent) by using dependency / constituency parsers. 

**AdNAN** is an effort to leverage this relational understanding for learning data patterns as an intermediate step to improving performance for downstream tasks, particularly in unsupervised settings like clustering.
