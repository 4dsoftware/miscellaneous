class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, target_dim, seq_length, d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.seq_length = seq_length
        self.d_model = d_model

        # Embedding layers for input and output
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.output_embedding = nn.Linear(target_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=seq_length)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )

        # Final output layer
        self.output_layer = nn.Linear(d_model, target_dim)

    def forward(self, X, y_in):
        # X: [batch_size, input_dim]
        # y_in: [batch_size, seq_length, target_dim]

        # Embed inputs
        X_embedded = self.input_embedding(X)  # [batch_size, d_model]
        X_embedded = X_embedded.unsqueeze(1).repeat(1, self.seq_length, 1)  # [batch_size, seq_length, d_model]

        # Embed outputs
        y_embedded = self.output_embedding(y_in)  # [batch_size, seq_length, d_model]

        # Add positional encoding
        X_encoded = self.positional_encoding(X_embedded)
        y_encoded = self.positional_encoding(y_embedded)

        # Transpose for transformer (seq_length, batch_size, d_model)
        X_encoded = X_encoded.permute(1, 0, 2)
        y_encoded = y_encoded.permute(1, 0, 2)

        # Pass through transformer
        output = self.transformer(X_encoded, y_encoded)  # [seq_length, batch_size, d_model]

        # Final linear layer
        output = self.output_layer(output)  # [seq_length, batch_size, target_dim]

        # Transpose back to (batch_size, seq_length, target_dim)
        output = output.permute(1, 0, 2)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Model initialization
input_dim = 20  # Number of features
output_dim = 10  # Number of targets
seq_length = 60  # Time points

model = TimeSeriesTransformer(
    input_dim=input_dim, 
    target_dim=output_dim, 
    seq_length=seq_length
)

# Example input
X = torch.randn(1000, input_dim)  # [batch_size, input_dim]
y_in = torch.zeros(1000, seq_length, output_dim)  # [batch_size, seq_length, target_dim]

# Forward pass
output = model(X, y_in)  # [batch_size, seq_length, target_dim]
print(output.shape)
