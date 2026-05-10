import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        max_idx = len(tokens) - context_length

        indices = torch.randint(0, max_idx, (batch_size,))
        random_starts = indices.tolist()

        X = []
        Y = []

        for i in indices:
            X.append(tokens[i : i+context_length])
            Y.append(tokens[i+1 : i+context_length+1])

        return X, Y
        
