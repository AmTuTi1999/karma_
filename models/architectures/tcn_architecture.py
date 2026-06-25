import torch.nn as nn

from utils.types.param_types import TCNModelParam


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

        # init
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, model_param: TCNModelParam):
        """
        input_channels: number of input features (channels)
        num_channels: list of channel sizes for each TemporalBlock, e.g. [64,64,128]
        kernel_size: conv kernel size (recommend 2 or 3)
        dropout: dropout probability
        output_size: number of output features per forecast step
        forecast_steps: how many future steps to predict (1 for one-step)
        """
        super().__init__()
        layers = []
        num_levels = len(model_param.num_channels)
        for i in range(num_levels):
            in_ch = (
                model_param.in_channels if i == 0 else model_param.num_channels[i - 1]
            )
            out_ch = model_param.num_channels[i]
            dilation = 2**i
            padding = (model_param.kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    model_param.kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=model_param.dropout,
                )
            )
        self.network = nn.Sequential(*layers)

        # A simple forecasting head: global (temporal) readout of last time-step features
        self.forecast_steps = model_param.forecast_steps
        self.output_size = model_param.output_size
        self.head = nn.Sequential(
            nn.Linear(model_param.num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(model_param.dropout),
            nn.Linear(128, model_param.output_size * model_param.forecast_steps),
        )

    def forward(self, x):
        """
        x: tensor shape (batch, input_channels, seq_len)
        returns: (batch, forecast_steps, output_size) if forecast_steps>1 else (batch, output_size)
        """
        y = self.network(x)  # (batch, channels, seq_len)
        last = y[:, :, -1]  # readout last time step: (batch, channels)
        out = self.head(last)  # (batch, output_size * forecast_steps)
        if self.forecast_steps > 1:
            out = out.view(x.size(0), self.forecast_steps, self.output_size)
        else:
            # Keep shape consistent with LSTM for one-step forecasting
            # so y shape can always be (batch, forecast_steps, output_size)
            out = out.view(x.size(0), 1, self.output_size)
        return out
