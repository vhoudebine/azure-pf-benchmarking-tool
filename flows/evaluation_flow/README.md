# Q&A Evaluation: f1-score

The Q&A f1-score evaluation flow will evaluate the Q&A Retrieval Augmented Generation systems using f1-score based on the word counts in predicted answer and ground truth.

## What you will learn

The f1-score evaluation flow allows you to determine the f1-score metric using number of common tokens between the normalized version of the ground truth and the predicted answer.

**F1-score**: Compute the f1-Score based on the tokens in the predicted answer and the ground truth.

 F1-score is a value in the range [0, 1]. 

## Prerequisites

- Data input: Evaluating the f1-Score metric requires you to provide data inputs including a ground truth and an answer. 

## Tools used in this flow
- Python tool