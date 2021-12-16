import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Field, BucketIterator

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint


def tokenize_per(text):
    return [tok.text for tok in text.split(" ")]


persian = Field(tokenize=tokenize_per, lower=True, init_token="<sos>", eos_token="<eos>")

url = 'http://github.com/amirmosio/validation.tar.gz'
path = torchtext.utils.download_from_url(url)
file = open(path, 'rb').read()

persian.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))

        encoder_states, (hidden, cell) = self.rnn(embedding)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
            self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))

        attention = self.softmax(energy)

        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        rnn_input = torch.cat((context_vector, embedding), dim=2)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        predictions = self.fc(outputs).squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(persian.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

num_epochs = 100
learning_rate = 3e-4
batch_size = 32

input_size_encoder = len(persian.vocab)
input_size_decoder = len(persian.vocab)
output_size = len(persian.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0

writer = SummaryWriter(f"runs/loss_plot")
step = 0
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)
encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)
decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = persian.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
sentence = (
    "ein boot mit mehreren männern darauf wird von einem großen"
    "pferdegespann ans ufer gezogen."
)
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, persian, persian, device, max_length=50
    )
    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

score = bleu(test_data[1:100], model, persian, persian, device)
print(f"Bleu score {score * 100:.2f}")
