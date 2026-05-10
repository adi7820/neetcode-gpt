import torch
import torch.nn as nn
# from torchtyping import TensorType

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Return your answer to 4 decimal places
        q = self.query_gen(embedded)
        k = self.key_gen(embedded)
        v = self.value_gen(embedded)

        score = q @ torch.transpose(k, 1, 2)
        context_len, attention_dim = k.shape[1], k.shape[2]

        score = score / (attention_dim ** 0.5)

        lower_traiangular = torch.tril(torch.ones(context_len, context_len))
        mask = lower_traiangular == 0

        score = score.masked_fill(mask, float('-inf'))
        score = nn.functional.softmax(score, dim=2)

        return torch.round(score @ v, decimals=4)
