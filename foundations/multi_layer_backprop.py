import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        linear_1 = np.dot(x, W1.T) + b1
        relu_1 = np.maximum(0, linear_1)
        linear_2 = np.dot(relu_1, W2.T) + b2

        loss = np.mean((linear_2 - y_true)**2)
        n = len(y_true)
        op_grad = 2*(linear_2 - y_true)/n

        dw2 = np.outer(op_grad, relu_1)
        db2 = op_grad

        da1 = np.dot(op_grad, W2)
        dz1 = da1 = da1 * (linear_1 > 0)
        dw1 = np.outer(dz1, x)
        dw1[dw1 == -0.0] = 0.0
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dw1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dw2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }
