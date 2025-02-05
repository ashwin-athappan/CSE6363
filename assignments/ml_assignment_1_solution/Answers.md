# Linear Regression

1. What are the pros and cons of using the normal equation to solve for the weights in linear regression as opposed to using gradient descent?


Normal Equation:

Pros:
 - Direct solution, no need for iterations.
 - Works well for small datasets.
 - No learning rate tuning required.

Cons:
 - Computationally expensive for large datasets (O(n³) due to matrix inversion).
 - Can be numerically unstable if XᵀX is ill-conditioned.


# Logistic Regression

1. Why is the softmax function used in multi-class logistic regression (Hint: the model itself produces logits)?

<p style="text-align: justify"> Softmax converts logits into probabilities that sum to 1, making it useful for multi-class classification. It ensures each class gets a proper probability score, unlike raw logits.</p>
