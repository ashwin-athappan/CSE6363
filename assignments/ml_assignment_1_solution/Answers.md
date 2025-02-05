# Linear Regression

1. What are the pros and cons of using the normal equation to solve for the weights in linear regression as opposed to using gradient descent?

```
Normal Equation:
- Pros:
  - No need to choose learning rate
  - No need to iterate
  - Directly computes the weights
  - No need to scale the features
- Cons:
  - Computationally expensive
  - Inverting the matrix can be slow
  - Not suitable for large datasets
```

# Logistic Regression

1. Why is the softmax function used in multi-class logistic regression (Hint: the model itself produces logits)?

<p style="text-align: justify"> The softmax function is used in multi-class logistic regression because it converts the logits into probabilities. The model produces logits which are the raw scores that are produced by the model. The softmax function converts these raw scores into probabilities. The probabilities are then used to make predictions.</p>
