import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Field

from utils import save_checkpoint, load_checkpoint


############################
## load and tokenize data ##
############################
def tokenize_per(text):
    return np.array([tok for tok in text.replace("\r\n", " ").split(" ")])


persian = Field(tokenize=tokenize_per, lower=True, init_token="<sos>", eos_token="<eos>")
file_name = 'ferdosi.txt'
# url = 'https://github.com/amirmosio/ferdosi-poem-generator/raw/main/ferdosi.txt'
# path = torch_text_utils.download_from_url(url)
poems_sentences = np.array(open(file_name, 'rb').read().decode('utf-8').split("\r\n"))
poems_tokens = np.array([tokenize_per(sent) for sent in poems_sentences])
seed_generator = torch.Generator().manual_seed(42)
train_poems_ids, test_poems_ids = torch.utils.data.random_split(range(len(poems_sentences) - 1),
                                                                [len(poems_sentences) - 11, 10],
                                                                generator=seed_generator)
persian.build_vocab(poems_tokens[train_poems_ids], max_size=10000, min_freq=2)


def convert_sentence_tokens_to_vectors(array):
    res = np.zeros((array.shape[0], len(persian.vocab)))
    for token_idx in range(array.shape[0]):
        vocab_index = persian.vocab[array[token_idx]]
        res[token_idx][vocab_index] = 1
    return res


# poems_vectors = convert_sentence_tokens_to_vectors(poems_tokens[0])

######################
### configuration ####
######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

num_epochs = 100
learning_rate = 0.01
batch_size = 60

input_size_encoder = len(persian.vocab)
input_size_decoder = len(persian.vocab)
output_size = len(persian.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0


######################
####### models #######
######################

class PoemsDataset:
    def __init__(self, data, ids):
        self.data = data
        self.length = len(ids) - 1
        self.ids = ids

    def get_batches(self, batch_size):
        num_batches = int(np.ceil(self.length / batch_size))

        indices = torch.randperm(self.length)

        first = np.array([self.data[i] for i in self.ids])[indices]
        second = np.array([self.data[i + 1] for i in self.ids])[indices]

        for batch in range(num_batches):
            f = first[batch * batch_size: min((batch + 1) * batch_size, self.length)]
            s = second[batch * batch_size: min((batch + 1) * batch_size, self.length)]
            yield batch, f, s


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(p)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))

        encoder_states, (hidden, cell) = self.rnn(embedding)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(p)
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

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
        batch_size = source.shape[0]
        target_len = target.shape[0]
        target_vocab_size = len(persian.vocab)
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs


writer = SummaryWriter(f"runs/loss_plot")

train_dataset = PoemsDataset(poems_sentences, train_poems_ids)
test_dataset = PoemsDataset(poems_sentences, test_poems_ids)

# initializing models
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(
    device)
model = Seq2Seq(encoder_net, decoder_net).to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = persian.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.train()
    for batch in train_dataset.get_batches(batch_size):
        inp_data = batch[1]
        target = batch[2]

        output = model(inp_data, target)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=epoch)

# score = bleu(test_data[1:100], model, persian, persian, device)
# print(f"Bleu score {score * 100:.2f}")
