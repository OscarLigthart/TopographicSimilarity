import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

from data import AgentVocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Receiver(nn.Module):
    """
    This class represents the Receiver agent in a referential game.
    It receives the message sent by the sender along with the target and some distractors.
    Its objective is to successfully predict the target based on the message it receives.

    """
    def __init__(
        self,
        vocab: AgentVocab,
        embedding_size: int = 64,
        hidden_size: int = 64,
        output_size: int = 64,
        cell_type: str = "lstm",
    ):
        super().__init__()

        print('checking')

        self.full_vocab_size = vocab.full_vocab_size

        self.vocab_size = vocab.vocab_size

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_type = cell_type

        if cell_type == "lstm":
            self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        else:
            raise ValueError(
                "Receiver case with cell_type '{}' is undefined".format(cell_type)
            )

        self.embedding = nn.Parameter(
            torch.empty((self.vocab_size, embedding_size), dtype=torch.float32)
        )

        self.output_module = nn.Identity()
        if self.output_size != self.hidden_size:
            self.output_module = nn.Linear(hidden_size, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)
        if type(self.rnn) is nn.LSTMCell:
            nn.init.xavier_uniform_(self.rnn.weight_ih)
            nn.init.orthogonal_(self.rnn.weight_hh)
            nn.init.constant_(self.rnn.bias_ih, val=0)
            nn.init.constant_(self.rnn.bias_hh, val=0)
            nn.init.constant_(
                self.rnn.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
            )

    def forward(self, messages):
        batch_size = messages.shape[0]

        messages = messages.float()

        emb = (
            torch.matmul(messages, self.embedding)
        )

        # initialize hidden
        h = torch.zeros([batch_size, self.hidden_size], device=device)
        if self.cell_type == "lstm":
            c = torch.zeros([batch_size, self.hidden_size], device=device)
            h = (h, c)

        # make sequence_length be first dim
        seq_iterator = emb.transpose(0, 1)
        for w in seq_iterator:
            h = self.rnn(w, h)

        if self.cell_type == "lstm":
            h = h[0]  # keep only hidden state, ditch cell state

        # convert hidden state to prediction
        out = self.output_module(h)

        return out, emb