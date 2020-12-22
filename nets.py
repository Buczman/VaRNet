import torch


class CAViaR(torch.nn.Module):
    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda')):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size1
        self.input_size = input_size
        self.output_size = 1  # in standard scenario only variance is returned
        self.device = device
        self.stateful = stateful

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size1, batch_first=True)

        self.linear1 = torch.nn.Linear(hidden_layer_size1, 64)
        self.linear2 = torch.nn.Linear(64, self.output_size)
        # self.linear3 = torch.nn.Linear(32, output_size)
        self.hidden_cell = None

        self.to(self.device)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.view(batch_size, input_seq.shape[1], -1))
        lin_out = self.linear2(lin_out)

        return lin_out[:, -1, 0]

    def init_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(1, batch_size, self.hidden_layer_size).to(self.device))


class GARCH(torch.nn.Module):
    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda')):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size1
        self.input_size = input_size
        self.output_size = 1  # in standard scenario only variance is returned
        self.device = device
        self.stateful = stateful
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size1, batch_first=True)

        self.linear1 = torch.nn.Linear(hidden_layer_size1, 64)
        self.linear2 = torch.nn.Linear(64, self.output_size)
        # self.linear3 = torch.nn.Linear(32, output_size)

        self.hidden_cell = None

        self.softplus = torch.nn.Softplus()
        self.to(self.device)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.view(batch_size, input_seq.shape[1], -1))
        lin_out = self.linear2(lin_out)

        return self.softplus(lin_out[:, -1, 0])

    def init_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(1, batch_size, self.hidden_layer_size).to(self.device))


class GARCHSkewedTStudent(GARCH):

    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda')):
        super().__init__(input_size, stateful, hidden_layer_size1, device)
        self.skewness = torch.autograd.Variable(torch.tensor(0., device=self.device), requires_grad=True)
        self.df = torch.autograd.Variable(torch.tensor(2.05, device=self.device), requires_grad=True)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.view(batch_size, input_seq.shape[1], -1))
        lin_out = self.linear2(lin_out)

        return torch.cat([
            self.softplus(lin_out[:, -1, 0]),
            self.df.unsqueeze(0),
            self.skewness.unsqueeze(0)
        ])


class GARCHTStudent(GARCH):

    def __init__(self, input_size=1, stateful=False, hidden_layer_size1=100, device=torch.device('cuda')):
        super().__init__(input_size, stateful, hidden_layer_size1, device)
        self.df = torch.autograd.Variable(torch.tensor(2.05), requires_grad=True, device=self.device)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        if not self.stateful:
            self.init_hidden(batch_size)
        else:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(batch_size, input_seq.shape[1], -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.view(batch_size, input_seq.shape[1], -1))
        lin_out = self.linear2(lin_out)

        return torch.cat([
            self.softplus(lin_out[:, -1, 0]),
            self.df.unsqueeze(0)
        ])
