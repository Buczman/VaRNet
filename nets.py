import torch
import torch.autograd.profiler as profiler
#----
from pytorch_lightning.core.lightning import LightningModule
from loss import caviar_loss, huber_loss

class CAViaRLightning(LightningModule):

    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda'), memory_size=1, learning_rate=1e-3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size1
        self.input_size = input_size
        self.output_size = 1  # in standard scenario only variance is returned
        self.stateful = stateful

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size1, batch_first=True)

        self.linear1 = torch.nn.Linear(hidden_layer_size1, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, self.output_size)
        self.hidden_cell = None
        self.relu = torch.nn.ReLU()

        self.learning_rate = learning_rate

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.relu(self.linear1(lstm_out[:, -1, :].view(batch_size, -1)))
        # lin_out = self.relu(self.linear1(lstm_out.contiguous().view(batch_size, -1)))
        lin_out = self.relu(self.linear2(lin_out))
        lin_out = self.linear3(lin_out)

        return lin_out

    def init_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device),
                            torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device))

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = huber_loss(y, out)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class CAViaR(torch.nn.Module):
    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda'), memory_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size1
        self.input_size = input_size
        self.output_size = 1  # in standard scenario only variance is returned
        self.device = device
        self.stateful = stateful

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size1, batch_first=True)

        self.linear1 = torch.nn.Linear(hidden_layer_size1, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, self.output_size)
        self.hidden_cell = None
        self.relu = torch.nn.ReLU()
        self.to(self.device)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.relu(self.linear1(lstm_out[:, -1, :].view(batch_size, -1)))
        # lin_out = self.relu(self.linear1(lstm_out.contiguous().view(batch_size, -1)))
        lin_out = self.relu(self.linear2(lin_out))
        lin_out = self.linear3(lin_out)

        return lin_out

    def init_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device),
                            torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device))


class GARCH(torch.nn.Module):
    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda'), memory_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size1
        self.input_size = input_size
        self.output_size = 1  # in standard scenario only variance is returned
        self.device = device
        self.stateful = stateful
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size1, batch_first=True)

        self.linear1 = torch.nn.Linear(hidden_layer_size1, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, self.output_size)

        self.hidden_cell = None
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        self.to(self.device)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.relu(self.linear1(lstm_out[:, -1, :].view(batch_size, -1)))
        lin_out = self.relu(self.linear2(lin_out))
        lin_out = self.linear3(lin_out)


        return self.softplus(lin_out)

    def init_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device),
                            torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device))


class GARCHSkewedTStudent(GARCH):

    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda'), memory_size=1):
        super().__init__(input_size, stateful, hidden_layer_size1, device, memory_size)
        self.skewness = torch.autograd.Variable(torch.tensor(0., device=self.device), requires_grad=True)
        self.df = torch.autograd.Variable(torch.tensor(2.05, device=self.device), requires_grad=True)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.relu(self.linear1(lstm_out[:, -1, :].view(batch_size, -1)))
        lin_out = self.relu(self.linear2(lin_out))
        lin_out = self.linear3(lin_out)

        return torch.cat([
            self.softplus(lin_out),
            self.df.unsqueeze(0).unsqueeze(0),
            self.skewness.unsqueeze(0).unsqueeze(0)
        ])


class GARCHTStudent(GARCH):

    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda'), memory_size=1):
        super().__init__(input_size, stateful, hidden_layer_size1, device, memory_size)
        self.df = torch.autograd.Variable(torch.tensor(2.05), requires_grad=True, device=self.device)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.contiguous().view(batch_size, -1))
        lin_out = self.linear2(lin_out)

        return torch.cat([
            self.softplus(lin_out),
            self.df.unsqueeze(0).unsqueeze(0)
        ])
