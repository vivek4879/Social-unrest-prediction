import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        #TO-DO
        #1. Initialize Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        #2. Initialize RNN layer
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          dropout=dropout,
                          bidirectional=bidirectional)

        #3. Initialize a fully connected layer with Linear transformation
        self.fc = nn.Linear(hidden_dim, output_dim)

        #4. Initialize Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        #text = [sent_len, batch_size]

        #TO-DO
        #1. Apply embedding layer that matches each word to its vector and apply dropout. Dim [sent_len, batch_size, emb_dim]
        emb = self.dropout(self.embedding(text))  # seq len, batch, emb dim

        packed_embedded = nn.utils.rnn.pack_padded_sequence(emb, text_lengths.to('cpu'))
        
        #2. Run the RNN along the sentences of length sent_len. #output = [sent len, batch size, hid dim * num directions]; #hidden = [num layers * num directions, batch size, hid dim]
        _, hidden = self.rnn(packed_embedded)

        #3. Get last forward (hidden[-1,:,:]) hidden layer and apply dropout
        hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)