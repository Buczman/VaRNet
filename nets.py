import torch


class CAViaR(torch.nn.Module):
    def __init__(self, input_size=1, batch_size=1, hidden_layer_size1=100, device=torch.device('cuda')):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size1
        self.input_size = input_size
        self.batch_size = batch_size
        self.output_size = 1  # in standard scenario only variance is returned
        self.device = device

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size1)

        self.linear1 = torch.nn.Linear(hidden_layer_size1, 64)
        self.linear2 = torch.nn.Linear(64, self.output_size)
        # self.linear3 = torch.nn.Linear(32, output_size)
        self.batch_size = batch_size

        self.hidden_cell = (torch.zeros(1, self.batch_size, self.hidden_layer_size),
                            torch.zeros(1, self.batch_size, self.hidden_layer_size))

        self.to(self.device)

    def forward(self, input_seq):
        self.hidden_cell = (torch.zeros(1, self.batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(1, self.batch_size, self.hidden_layer_size).to(self.device))

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), self.batch_size, -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.view(len(input_seq), self.batch_size, -1))
        lin_out = self.linear2(lin_out)

        return lin_out[-1][0]


class GARCH(torch.nn.Module):
    def __init__(self, input_size=1, batch_size=1, hidden_layer_size1=100, device=torch.device('cuda')):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size1
        self.input_size = input_size
        self.batch_size = batch_size
        self.output_size = 1  # in standard scenario only variance is returned
        self.device = device

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size1)

        self.linear1 = torch.nn.Linear(hidden_layer_size1, 64)
        self.linear2 = torch.nn.Linear(64, self.output_size)
        # self.linear3 = torch.nn.Linear(32, output_size)

        self.hidden_cell = (torch.zeros(self.input_size, self.batch_size, self.hidden_layer_size),
                            torch.zeros(self.input_size, self.batch_size, self.hidden_layer_size))

        self.softplus = torch.nn.Softplus()
        self.to(self.device)

    def forward(self, input_seq):
        self.hidden_cell = (torch.zeros(self.input_size1, self.batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(self.input_size, self.batch_size, self.hidden_layer_size).to(self.device))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.view(len(input_seq), -1))
        lin_out = self.linear2(lin_out)

        return self.softplus(lin_out[-1][0])


class GARCHSkewedTStudent(GARCH):

    def __init__(self, input_size=1, batch_size=1, hidden_layer_size1=100, device=torch.device('cuda')):
        super().__init__(input_size, batch_size, hidden_layer_size1, device)
        self.skewness = torch.autograd.Variable(torch.tensor(0., device=self.device), requires_grad=True)
        self.df = torch.autograd.Variable(torch.tensor(2.05, device=self.device), requires_grad=True)

    def forward(self, input_seq):
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(self.device))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.view(len(input_seq), -1))
        lin_out = self.linear2(lin_out)

        return torch.stack([
            self.softplus(lin_out[-1][0]),
            self.df,
            self.skewness
        ])


class GARCHTStudent(GARCH):

    def __init__(self, input_size=1, batch_size=1, hidden_layer_size1=100, device=torch.device('cuda')):
        super().__init__(input_size, batch_size, hidden_layer_size1, device)
        self.df = torch.autograd.Variable(torch.tensor(2.05), requires_grad=True, device=self.device)

    def forward(self, input_seq):
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(self.device))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        lin_out = self.linear1(lstm_out.view(len(input_seq), -1))
        lin_out = self.linear2(lin_out)

        return torch.stack([
            self.softplus(lin_out[-1][0]),
            self.df
        ])
