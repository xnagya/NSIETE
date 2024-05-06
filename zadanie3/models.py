# %% [markdown]
# # __Neural network models__

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc

# %% [markdown]
# ## Custom LSTM Networks

# %%
class LSTM_custom(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, drop):
        torch.set_default_dtype(torch.float32)
        super().__init__()
        
        self.bidir = bidirectional
        self.gates_size = (hidden_size * 2) if bidirectional else (hidden_size * 4)
        self.eps = 1e-16

        assert (self.gates_size % 4 == 0), f"Wrong shape for lstm! Must be divisible by 4, but got {hidden_size} instead."
        
        # Network parameters
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, self.gates_size))
        self.weight_hh = nn.Parameter(torch.Tensor(self.gates_size // 4, self.gates_size))
        self.bias_ih = nn.Parameter(torch.Tensor(self.gates_size))
        self.bias_hh = nn.Parameter(torch.Tensor(self.gates_size))

        # Reverse network parameters
        if bidirectional:
            self.weight_Rih = nn.Parameter(torch.Tensor(input_size, self.gates_size))
            self.weight_Rhh = nn.Parameter(torch.Tensor(self.gates_size // 4, self.gates_size))
            self.bias_Rih = nn.Parameter(torch.Tensor(self.gates_size))
            self.bias_Rhh = nn.Parameter(torch.Tensor(self.gates_size))
        
        # Add dropout layer
        self.dropout = nn.Dropout(drop)

    # Forward pass for LSTM layer
    def forward(self, input, state, device):
        _ , seq_length, _ = input.shape

        backward_state = state
        
        # Forward pass for each word in sequence
        layer_output = []

        for i in range(0, seq_length, 1):
            # Get word from sequence
            x = input[:,i,:].to(device)

            # Forward pass for word
            x, state = self.forward_cell(x, state)

            # Save results 
            layer_output.append(x.unsqueeze(0))

        layer_output = LSTM_custom.join_layer_output(layer_output)

        # Reverse forward pass for each word in sequence
        if self.bidir:
            reverse_output = []

            for i in range(seq_length - 1, -1, -1):
                # Get word from sequence
                x = input[:,i,:].to(device)

                # Forward pass for word
                x, backward_state = self.reverse_cell(x, backward_state)

                # Save results 
                reverse_output.append(x.unsqueeze(0))
            
            reverse_output = LSTM_custom.join_layer_output(reverse_output)

            # Join results from both directions
            layer_output = torch.cat((layer_output, reverse_output), dim=2)
            state = torch.cat((state + backward_state), dim=1)

        return layer_output, state

    @abc.abstractmethod
    def initial_state(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward_cell(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def reverse_cell(self):
        raise NotImplementedError
    
    def join_layer_output(outputs: list[torch.Tensor]):
        # Join results
        t = torch.cat(outputs, dim=0)
        # Reshape to batch_size, sequence_length, hidden_size
        t = t.transpose(0, 1).contiguous()
        return t

# %%
class LSTM_basic(LSTM_custom):
    def __init__(self, input_size, hidden_size, params: dict, drop = 0):
        # Config parameters
        try:
            bidir = params['bidirectional']
        except KeyError as e:
            raise Exception(f'Parameter "{e.args[0]}" NOT found!')
        
        super().__init__(input_size, hidden_size, bidir, drop)
        
    def initial_state(self, batch_size):
        h0 = torch.zeros(batch_size, self.gates_size // 4)
        c0 = torch.zeros(batch_size, self.gates_size // 4)
        return (h0, c0)
    
    # Computes forward for one timestep (one word of sequence)
    def forward_cell(self, xt, state):
        # Load current state 
        ht, ct = state

        # Forward pass
        gates = torch.mm(xt, self.weight_ih) + self.bias_ih + torch.mm(ht, self.weight_hh) + self.bias_hh

        # Devide tensor into gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim = 1)

        # Compute state of each gate
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        # Compute new lstm state
        ct = torch.mul(ct, forgetgate) +  torch.mul(ingate, cellgate)     
        ht = torch.mul(outgate, F.tanh(ct))

        # Dropout(x), state
        return self.dropout(ht), (ht, ct)
    
    # Computes forward for one timestep (using reverse weights)
    def reverse_cell(self, xt, state):
        # Load current state 
        ht, ct = state

        # Forward pass
        gates = torch.mm(xt, self.weight_Rih) + self.bias_Rih + torch.mm(ht, self.weight_Rhh) + self.bias_Rhh

        # Devide tensor into gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim = 1)

        # Compute state of each gate
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        # Compute new lstm state
        ct = torch.mul(ct, forgetgate) +  torch.mul(ingate, cellgate)     
        ht = torch.mul(outgate, F.tanh(ct))

        # Dropout(x), state
        return self.dropout(ht), (ht, ct)

# %%
class LSTM_momentum(LSTM_custom):
    def __init__(self, input_size, hidden_size, params: dict, drop = 0):
        # Config parameters
        try:
            bidir = params['bidirectional']
            # Momentum cell hyperparameters
            self.mu = params['momentum']
            self.s = params['stepsize']
        except KeyError as e:
            raise Exception(f'Parameter "{e.args[0]}" NOT found!')
        
        super().__init__(input_size, hidden_size, bidir, drop)
        
    def initial_state(self, batch_size):
        h0 = torch.zeros(batch_size, self.gates_size // 4)
        c0 = torch.zeros(batch_size, self.gates_size // 4)
        v0 = torch.zeros(batch_size, self.gates_size)
        return (h0, c0, v0)

    # Computes forward for one timestep (one word of sequence)
    def forward_cell(self, xt, state):
        # Load current state 
        ht, ct, vt = state

        # Forward pass
        vt = self.mu * vt + self.s * (torch.mm(xt, self.weight_ih) + self.bias_ih)
        gates = vt + torch.mm(ht, self.weight_hh) + self.bias_hh
        #print(f"gates = {gates.shape}")

        # Devide tensor into gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim = 1)

        # Compute state of each gate
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        # Compute new lstm state
        ct = torch.mul(ct, forgetgate) +  torch.mul(ingate, cellgate)     
        ht = torch.mul(outgate, F.tanh(ct))

        # Dropout(x), state
        return self.dropout(ht), (ht, ct, vt)
    
    # Computes forward for one timestep (using reverse weights)
    def reverse_cell(self, xt, state):
        # Load current state 
        ht, ct, vt = state

        # Forward pass
        vt = self.mu * vt + self.s * (torch.mm(xt, self.weight_Rih) + self.bias_Rih)
        gates = vt + torch.mm(ht, self.weight_Rhh) + self.bias_Rhh
        #print(f"gates = {gates.shape}")

        # Devide tensor into gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim = 1)

        # Compute state of each gate
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        # Compute new lstm state
        ct = torch.mul(ct, forgetgate) +  torch.mul(ingate, cellgate)     
        ht = torch.mul(outgate, F.tanh(ct))

        # Dropout(x), state
        return self.dropout(ht), (ht, ct, vt)

# %%
class LSTM_adam(LSTM_custom):
    def __init__(self, input_size, hidden_size, params: dict, drop = 0):
        # Config parameters
        try:
            bidir = params['bidirectional']
            # Adam cell hyperparameters
            self.mu = params['momentum']
            self.s = params['stepsize']
            self.b = params['rnn_beta']
        except KeyError as e:
            raise Exception(f'Parameter "{e.args[0]}" NOT found!')
        
        super().__init__(input_size, hidden_size, bidir, drop)
        
    def initial_state(self, batch_size):
        h0 = torch.zeros(batch_size, self.gates_size // 4)
        c0 = torch.zeros(batch_size, self.gates_size // 4)
        v0 = torch.zeros(batch_size, self.gates_size)
        m0 = torch.zeros(batch_size, self.gates_size)
        return (h0, c0, v0, m0)

    # Computes forward for one timestep (one word of sequence)
    def forward_cell(self, xt, state):
        # Load current state 
        ht, ct, vt, mt = state

        # Forward pass
        grad = torch.mm(xt, self.weight_ih) + self.bias_ih
        vt = self.mu * vt + self.s * grad
        mt = self.b * mt + (1 - self.b) * (grad * grad)
        gates = (vt / (torch.sqrt(mt) + self.eps)) + torch.mm(ht, self.weight_hh) + self.bias_hh
        #print(f"gates = {gates.shape}")

        # Devide tensor into gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim = 1)

        # Compute state of each gate
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        # Compute new lstm state
        ct = torch.mul(ct, forgetgate) +  torch.mul(ingate, cellgate)     
        ht = torch.mul(outgate, F.tanh(ct))

        # Dropout(x), state
        return self.dropout(ht), (ht, ct, vt, mt)
    
    # Computes forward for one timestep (using reverse weights)
    def reverse_cell(self, xt, state):
        # Load current state 
        ht, ct, vt, mt = state

        # Forward pass
        grad = torch.mm(xt, self.weight_Rih) + self.bias_Rih
        vt = self.mu * vt + self.s * grad
        mt = self.b * mt + (1 - self.b) * (grad * grad)
        gates = (vt / (torch.sqrt(mt) + self.eps)) + torch.mm(ht, self.weight_Rhh) + self.bias_Rhh
        #print(f"gates = {gates.shape}")

        # Devide tensor into gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim = 1)

        # Compute state of each gate
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        # Compute new lstm state
        ct = torch.mul(ct, forgetgate) +  torch.mul(ingate, cellgate)     
        ht = torch.mul(outgate, F.tanh(ct))

        # Dropout(x), state
        return self.dropout(ht), (ht, ct, vt, mt)

# %%
ALL_RNNtypes = ["simple", "lstm", "lstm_M", "lstm_A"]

class RNN(nn.Module):
    def __init__(self, rnn_type, embedding_path, params: dict):
        torch.set_default_dtype(torch.float32)
        super().__init__()

        assert rnn_type in ALL_RNNtypes, f"RNN type '{rnn_type} 'is NOT supported."
        self.rnn_type = rnn_type

        # Config parameters
        try:
            # Embedding layer
            drop_embed = params['embedding_dropout']
            self.pad_idx = params['padding_index']
            vocabulary_size = params['vocab_size']

            # RNN layer
            self.layer_count = params['rnn_layers']
            drop_rnn = params['rnn_dropout'] 
            hidden_size = params['hidden_features']
            bidir = params['bidirectional']
        except KeyError as e:
            raise Exception(f'Parameter "{e.args[0]}" NOT found!')
        
        # Load embedding matrix
        embed_matrix = torch.from_numpy(np.load(embedding_path)).type(torch.float32)
        matrix_size, embedding_dims = embed_matrix.shape

        assert (matrix_size == vocabulary_size), f"Size of embedding matrix '{matrix_size}' is different from vocabulary size '{vocabulary_size}'!"

        # Encoder layer = encodes indices of words to embedding vectors
        self.encoder = nn.Embedding.from_pretrained (
            embeddings = embed_matrix, 
            freeze = False, 
            padding_idx = self.pad_idx
            )

        # Dropout layer -> drops embedding features
        self.dropout = nn.Dropout(drop_embed)

        # Initialize RNN layers
        self.rnns = nn.ModuleList()

        match self.rnn_type:
            # RNN - built_in
            case "simple":
                # Reduce hidden_size to half if bidirectional
                output_size = (hidden_size // 2) if bidir else hidden_size

                self.rnns.append(nn.RNN(
                    input_size = embedding_dims, 
                    hidden_size = output_size, 
                    num_layers = self.layer_count, 
                    bidirectional = bidir,
                    dropout = drop_rnn, 
                    batch_first = True
                ))

            # LSTM
            case "lstm":
                for i in range(self.layer_count):
                    # First layer
                    if (i == 0):
                        self.rnns.append(LSTM_basic(embedding_dims, hidden_size, params, drop_rnn))
                    # Last layer, no dropout
                    elif (i == self.layer_count - 1):
                        self.rnns.append(LSTM_basic(hidden_size, hidden_size, params))
                    # Other layers
                    else:
                        self.rnns.append(LSTM_basic(hidden_size, hidden_size, params, drop_rnn))               

            # LSTM - momentum
            case "lstm_M":
                for i in range(self.layer_count):
                    # First layer
                    if (i == 0):
                        self.rnns.append(LSTM_momentum(embedding_dims, hidden_size, params, drop_rnn))
                    # Last layer, no dropout
                    elif (i == self.layer_count - 1):
                        self.rnns.append(LSTM_momentum(hidden_size, hidden_size, params))
                    # Other layers
                    else:
                        self.rnns.append(LSTM_momentum(hidden_size, hidden_size, params, drop_rnn))    

            # LSTM - momentum ADAM
            case "lstm_A":
                for i in range(self.layer_count):
                    # First layer
                    if (i == 0):
                        self.rnns.append(LSTM_adam(embedding_dims, hidden_size, params, drop_rnn))
                    # Last layer, no dropout
                    elif (i == self.layer_count - 1):
                        self.rnns.append(LSTM_adam(hidden_size, hidden_size, params))
                    # Other layers
                    else:
                        self.rnns.append(LSTM_adam(hidden_size, hidden_size, params, drop_rnn))    

        # Decoder layer = output layer for network
        self.decoder = nn.Linear(hidden_size, vocabulary_size)

        # Initialize network weights (embedding is pretrained)
        self.init_weights()
                 
    def init_weights(self):
        for model in self.rnns:
            for name, par in model.named_parameters(): 
                # Initialize weights
                if 'weight' in name:
                    nn.init.xavier_uniform_(par)

                # Initialize bias
                elif 'bias' in name:
                    par.data.fill_(0)

        # Output layer initialization
        for name, par in self.decoder.named_parameters():
            # Initialize weights
            if 'weight' in name:
                nn.init.xavier_uniform_(par)

            # Initialize bias
            if 'bias' in name:
                par.data.fill_(0)

    # Remove as many as possible zeroes (padding) from tensor
    # t.shape = (batch_size, sequence_length)
    def reduce_tensor(t: torch.tensor, pad_index):
        zeros_count = (t == float(pad_index)).sum(dim=1)
        #print(zeros_count)
        
        lowest_count = torch.min(zeros_count)

        new_length = t.shape[1] - lowest_count
        return t[:,0 : new_length]

    # Input shape = (batch_size, sequence_length)
    def forward(self, input, indexes: torch.tensor, device):
        # Reduce input sequence length
        input = RNN.reduce_tensor(input, self.pad_idx)

        batch_size, sequence_length  = input.shape

        """
        for row in indexes:
            for val in row:
                assert (val < sequence_length), f"Wrong index position '{val}' for input with length {sequence_length}."
        """

        # Embedding 
        input = self.encoder(input)

        # Dropout 
        input = self.dropout(input)

        # Clear layer states
        self.states = []

        # Custom LSTM -> multiple states (one per layer)
        if (self.rnn_type != "simple"):
            self.state = []

            # Initiate layer states
            for layer in self.rnns:
                self.state.append(layer.initial_state(batch_size))

            # Compute RNN forward for each layer
            for i, layer in enumerate(self.rnns):
                #print(lstm)
                current_state = self.state[i]
                input, current_state = layer.forward(input, current_state, device)
                self.state[i] = current_state

        # RNN -> states of layers are in one tensor
        else:
            # Initiate RNN state
            self.state = None

            for layer in self.rnns:
                input, self.state = layer.forward(input, self.state)

        # Extract missing words based od indexes
        output = torch.empty(input.shape[0], indexes.shape[1], input.shape[2]).to(device)
        for i in range(batch_size):
            row = indexes[i,:].tolist()
            words = input[i,row,:].unsqueeze(dim=0)
            torch.cat((output, words))

        # Create output with linear layer
        return self.decoder(output)
    

# %%
"""
from net_config import *
embedding_path = "embedding_matrix.npy"

# Input tensor
batch_size = 8
seq_length = 10

test_input = torch.randint(0, 7054, (batch_size, seq_length))
test_indexes = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]])

print(f"INPUT SHAPE = {test_input.shape}")
print(f"INDEXES SHAPE = {test_indexes.shape}")

net = RNN("simple", embedding_path, config_to_dict(config_NN))

output = net.forward(test_input, test_indexes)
print(f"OUTPUT SHAPE = {output.shape}")

output = torch.argmax(output, dim=2)
print(f"HARDMAX SHAPE = {output.shape}")
print(output)
"""


