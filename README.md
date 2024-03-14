# AG_NEWS Text Classification
## Overview
This project outlines the development of a Neural Text Classifier designed to categorize news articles into predefined classes based on their content. Utilizing the AG NEWS dataset, the classifier follows a systematic approach to text classification by transforming sentences into numerical data, applying neural network models, and optimizing for accuracy.

## Methodology
1. Data Preprocessing:
- Vocabulary Construction: Iterate over the dataset to build a vocabulary of unique words across all sentences.
- One-hot Encoding: Convert each word to a one-hot vector representation based on a unique integer index assigned from the vocabulary. For example, with a vocabulary size of 10, the word "the" indexed at 3 is represented as (0,0,1,0,0,0,0,0,0,0).
- Integer Index Mapping: Transform sentences into sequences of integer indices corresponding to the words' positions in the vocabulary.
- Batching and Padding: To accommodate variable sentence lengths within fixed-size batches, apply padding tokens to ensure uniform dimensionality. For instance, sentences ["a b c", "a b c d e"] in a batch would be padded and represented as [[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]] for consistent tensor dimensions.
2. Model Training:
- Batch Transformation: Feed batches of padded integer-indexed sentences to the model, translating them into tensors with dimensions corresponding to batch size and maximum sentence length in the batch.
- One-hot Vector Conversion: Expand the input tensors to a (Batch, Max_Length, Vocab_Size) dimensionality by converting integer indices to one-hot vectors.
- Feature Aggregation: Average the features across the sentence length to condense each sentence into a single vector of dimension (Batch, Vocab_Size).
- Neural Network Processing: Pass the aggregated feature vectors through a series of linear and non-linear layers. The network outputs logits for multiclass classification without applying a softmax function.
3. Optimization and Evaluation:
- Objective: Optimize the neural network to improve classification accuracy using a suitable loss function and optimization algorithm.
- Performance Metrics: Evaluate the model's performance on both training and validation datasets to assess its accuracy and generalization capability.


## Dataset
AG_NEWS is a dataset commonly used in text classification and natural language processing tasks. It stands for "AG's News Topic Classification Dataset" and is part of the AG's corpus of news articles. The dataset is designed for benchmarking machine learning models on the task of categorizing text into predefined news categories. please see the https://huggingface.co/datasets/ag_news for more details.

