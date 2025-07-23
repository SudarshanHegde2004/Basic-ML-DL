import numpy as np

class Attention:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(1, hidden_size) * 0.01
        
    def forward(self, query, keys, values, mask=None):
        # query shape: (batch_size, hidden_size)
        # keys shape: (batch_size, seq_len, hidden_size)
        # values shape: (batch_size, seq_len, hidden_size)
        
        # Calculate attention scores
        scores = np.dot(np.dot(query, self.W), keys.transpose(0, 2, 1))
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax to get attention weights
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Calculate context vector
        context = np.sum(attention_weights.reshape(-1, 1) * values, axis=1)
        
        return context, attention_weights

class MultiHeadAttention:
    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Initialize weights for each head
        self.W_q = [np.random.randn(hidden_size, self.head_size) * 0.01 for _ in range(num_heads)]
        self.W_k = [np.random.randn(hidden_size, self.head_size) * 0.01 for _ in range(num_heads)]
        self.W_v = [np.random.randn(hidden_size, self.head_size) * 0.01 for _ in range(num_heads)]
        self.W_o = np.random.randn(hidden_size, hidden_size) * 0.01
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Initialize output
        multi_head_output = []
        
        for head in range(self.num_heads):
            # Project query, key, and value for current head
            q = np.dot(query, self.W_q[head])
            k = np.dot(key, self.W_k[head])
            v = np.dot(value, self.W_v[head])
            
            # Calculate attention scores
            scores = np.dot(q, k.transpose(0, 2, 1)) / np.sqrt(self.head_size)
            
            # Apply mask if provided
            if mask is not None:
                scores = np.where(mask == 0, -1e9, scores)
            
            # Apply softmax
            attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            
            # Calculate head output
            head_output = np.dot(attention_weights, v)
            multi_head_output.append(head_output)
        
        # Concatenate all heads and project
        concat_output = np.concatenate(multi_head_output, axis=-1)
        final_output = np.dot(concat_output, self.W_o)
        
        return final_output
