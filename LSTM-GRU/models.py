import torch
import torch.nn as nn

class LSTM_PyTorch_Model(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, out_dim=10, num_layers=1, dropout=0.0, hc_mode="zero_init", rnn_class=None):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim =  hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_layers = num_layers
        
        if rnn_class is None:
            rnn_class = nn.LSTM
        
        assert hc_mode in ["zero_learned", "random_learned", "zero_init", "random_init"]
        self.hc_mode = hc_mode
        
        if "zero_learned" == hc_mode:
            self.h = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim).requires_grad_())
            self.c = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim).requires_grad_())
        elif "random_learned" == hc_mode:
            self.h = nn.Parameter(torch.randn(num_layers, 1, hidden_dim).requires_grad_())
            self.c = nn.Parameter(torch.randn(num_layers, 1, hidden_dim).requires_grad_())      
        
        # Embedding
        self.encoder = nn.Linear(in_features=input_dim, out_features=embed_dim)
        
        # RNN
        self.rnn  =  rnn_class(input_size=embed_dim,
                               hidden_size=hidden_dim, 
                               num_layers=num_layers, 
                               batch_first=True,
                               dropout=dropout)
        # Classifier
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=out_dim)
    
    
    def forward(self, x):
        batch_size, num_channels, rows, cols = x.shape
        
        # Assuming that data has single channel
        assert num_channels == 1
        
        h, c = self.init_state(batch_size=batch_size, device=x.device) 
        
        # (batch_size, 1, rows, cols) -> (batch_size, rows, cols)
        x = x.view(batch_size*num_channels*rows, cols)
        
        # (batch_size, rows, cols) -> (batch_size*rows, embed_dim)
        embeddings = self.encoder(x)
        
        # (batch_size*rows, embed_dim) -> (batch_size, rows, embed_dim)
        embeddings = embeddings.view(batch_size, rows, self.embed_dim)
        
        # (batch_size, rows, embed_dim) -> (batch_size, sequence_length(rows), hidden_dim)
        rnn_out, (h_out, c_out) = self.rnn(embeddings, (h,c))
        
        # (batch_size, sequence_length(rows), hidden_dim) -> (batch_size, out_dim)
        output = self.classifier(rnn_out[:, -1, :])  # feeding only output of last sequences

        return output
    
        
    def init_state(self, batch_size, device):
        if(self.hc_mode == "zero_init"):
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        elif(self.hc_mode == "random_init"):
            h = torch.randn(self.num_layers, batch_size, self.hidden_dim)
            c = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        elif(self.hc_mode == "zero_learned" or self.hc_mode == "random_learned"):
            h = self.h.repeat(1, batch_size, 1)
            c = self.c.repeat(1, batch_size, 1)

        return h.to(device), c.to(device)

#################################################################################################################

# Method abstracts are copied, because we aim using the same DL model only by replacing the original LSTM class

# https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None):
        super().__init__()
        
        assert bias == True # not implemented
        assert device == None # not implemented
        
        self.W_ii = nn.Parameter(torch.zeros((input_size, hidden_size)), requires_grad=True)
        self.W_hi = nn.Parameter(torch.zeros((hidden_size, hidden_size)), requires_grad=True)
        self.W_if = nn.Parameter(torch.zeros((input_size, hidden_size)), requires_grad=True)
        self.W_hf = nn.Parameter(torch.zeros((hidden_size, hidden_size)), requires_grad=True)
        self.W_ig = nn.Parameter(torch.zeros((input_size, hidden_size)), requires_grad=True)
        self.W_hg = nn.Parameter(torch.zeros((hidden_size, hidden_size)), requires_grad=True)
        self.W_io = nn.Parameter(torch.zeros((input_size, hidden_size)), requires_grad=True)
        self.W_ho = nn.Parameter(torch.zeros((hidden_size, hidden_size)), requires_grad=True)

        self.b_ii = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_hi = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_if = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_hf = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_ig = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_hg = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_io = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_ho = nn.Parameter(torch.zeros(1), requires_grad=True)
        
        nn.init.xavier_uniform_(self.W_ii)
        nn.init.xavier_uniform_(self.W_hi)
        nn.init.xavier_uniform_(self.W_if)
        nn.init.xavier_uniform_(self.W_hf)
        nn.init.xavier_uniform_(self.W_ig)
        nn.init.xavier_uniform_(self.W_hg)
        nn.init.xavier_uniform_(self.W_io)
        nn.init.xavier_uniform_(self.W_ho)
        
    def forward(self, *xhc):
        x, (h, c) = xhc
        
        i = torch.sigmoid( x @ self.W_ii + self.b_ii + h @ self.W_hi + self.b_hi )
        f = torch.sigmoid( x @ self.W_if + self.b_if + h @ self.W_hf + self.b_hf )
        g = torch.tanh( x @ self.W_ig + self.b_ig + h @ self.W_hg + self.b_hg )
        o = torch.sigmoid( x @ self.W_io + self.b_io + h @ self.W_ho + self.b_ho )
        c_new = torch.mul(f,c) + torch.mul(i,g)
        h_new = torch.mul(o, torch.tanh(c_new))
        
        return h_new, c_new


# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, device=None):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.device = device
        
        self.lstm_cell = nn.ModuleList()
        self.lstm_cell.append( MyLSTMCell(input_size, hidden_size, bias=bias) )
        for i in range(num_layers-1):
            self.lstm_cell.append( MyLSTMCell(hidden_size, hidden_size, bias=bias) )
        
        assert batch_first == True # not implemented
        assert dropout == 0.0 # not implemented
        assert device == None # not implemented

    def forward(self, *xhc):
        x, (h, c) = xhc
        
        x_ = x
        new_h = []
        new_c = []
        for layer in range(self.num_layers):
            output = []

            h_, c_ = h[layer], c[layer]
            
            for seq in range(x.shape[1]):
                h_, c_ = self.lstm_cell[layer](x_[:,seq,:], (h_, c_))
                output.append(h_)
                
            output = torch.stack(output, dim=1)
            x_ = output # this layers output will be the input for the next layer if num_layers>1
            
            new_h.append(h_)
            new_c.append(c_)
        
        new_h = torch.stack(new_h)
        new_c = torch.stack(new_c)
        return output, (new_h, new_c)

##################################################################################################

class OurGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        super(OurGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        
        k = torch.tensor(1/self.hidden_size)
        
        self.w_z = torch.nn.Parameter(torch.rand(self.hidden_size+self.input_size, self.hidden_size) * 2 * torch.sqrt(k) - torch.sqrt(k), requires_grad=True)
        
        self.w_r = torch.nn.Parameter(torch.rand(self.hidden_size+self.input_size, self.hidden_size) * 2 * torch.sqrt(k) - torch.sqrt(k), requires_grad=True)
        
        self.w = torch.nn.Parameter(torch.rand(self.hidden_size+self.input_size, self.hidden_size) * 2 * torch.sqrt(k) - torch.sqrt(k), requires_grad=True)
        
        self.bias_z = torch.nn.Parameter(torch.rand(1)*2*torch.sqrt(k) - torch.sqrt(k), requires_grad=True)
        self.bias_r = torch.nn.Parameter(torch.rand(1)*2*torch.sqrt(k) - torch.sqrt(k), requires_grad=True)
        self.bias_w = torch.nn.Parameter(torch.rand(1)*2*torch.sqrt(k) - torch.sqrt(k), requires_grad=True) 
        
    def forward(self, x, h):
        
        z_t = torch.sigmoid(torch.cat((h,x), 1) @ self.w_z)
        r_t = torch.sigmoid(torch.cat((h,x), 1) @ self.w_r)
        h_hat_t = torch.tanh(torch.cat((torch.mul(r_t, h), x), 1) @ self.w)
        h_t = (1-z_t) * h + z_t * h_hat_t
        return h_t

class OurGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=False):
        super(OurGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        self.gru_cell = nn.ModuleList()
        self.gru_cell.append( OurGRUCell(input_size, hidden_size, bias=bias) )
        for i in range(num_layers-1):
            self.gru_cell.append( OurGRUCell(hidden_size, hidden_size, bias=bias) )
        
        assert batch_first == True
        assert dropout == 0.0

    def forward(self, *xhc):
        x, h = xhc
        
        x_ = x
        new_h = []
        for layer in range(self.num_layers):
            output = []       
            
            h_ = h[layer]
            
            for seq in range(x.shape[1]):
                h_= self.gru_cell[layer](x_[:,seq,:], (h_))
                output.append(h_)
                
            output = torch.stack(output, dim=1)
            x_ = output
            
            new_h.append(h_)
        
        new_h = torch.stack(new_h).reshape((self.num_layers, len(x), self.hidden_size))
        return output, new_h


class GRU_PyTorch_Model(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, out_dim=10, num_layers=1, dropout=0.0, hc_mode="zero_init", rnn_class=None):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim =  hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_layers = num_layers
        
        if rnn_class is None:
            rnn_class = nn.LSTM
        
        assert hc_mode in ["zero_learned", "random_learned", "zero_init", "random_init"]
        self.hc_mode = hc_mode
        
        if "zero_learned" == hc_mode:
            self.h = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim).requires_grad_())
            self.c = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim).requires_grad_())
        elif "random_learned" == hc_mode:
            self.h = nn.Parameter(torch.randn(num_layers, 1, hidden_dim).requires_grad_())
            self.c = nn.Parameter(torch.randn(num_layers, 1, hidden_dim).requires_grad_())      
        
        # Embedding
        self.encoder = nn.Linear(in_features=input_dim, out_features=embed_dim)
        
        # RNN
        self.rnn  =  rnn_class(input_size=embed_dim,
                               hidden_size=hidden_dim, 
                               num_layers=num_layers, 
                               batch_first=True,
                               dropout=dropout)
        # Classifier
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=out_dim)
    
    
    def forward(self, x):
        batch_size, num_channels, rows, cols = x.shape
        
        # Assuming that data has single channel
        assert num_channels == 1
        
        h = self.init_state(batch_size=batch_size, device=x.device) 
        
        # (batch_size, 1, rows, cols) -> (batch_size, rows, cols)
        x = x.view(batch_size*num_channels*rows, cols)
        
        # (batch_size, rows, cols) -> (batch_size*rows, embed_dim)
        embeddings = self.encoder(x)
        
        # (batch_size*rows, embed_dim) -> (batch_size, rows, embed_dim)
        embeddings = embeddings.view(batch_size, rows, self.embed_dim)
        
        # (batch_size, rows, embed_dim) -> (batch_size, sequence_length(rows), hidden_dim)
        rnn_out, h_out = self.rnn(embeddings, h)
        
        # (batch_size, sequence_length(rows), hidden_dim) -> (batch_size, out_dim)
        output = self.classifier(rnn_out[:, -1, :])  # feeding only output of last sequences

        return output
    
        
    def init_state(self, batch_size, device):
        if(self.hc_mode == "zero_init"):
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        elif(self.hc_mode == "random_init"):
            h = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        elif(self.hc_mode == "zero_learned" or self.hc_mode == "random_learned"):
            h = self.h.repeat(1, batch_size, 1)

        return h.to(device)