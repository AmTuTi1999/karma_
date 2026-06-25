import torch.nn as nn

from utils.types.param_types import LSTMModelParam


class LSTM(nn.Module):
    """
    Stacked LSTM with LayerNorm and dropout for multivariate time-series
    multi-step regression forecasting.

    Args:
      input_size: number of input features (D)
      hidden_size: LSTM hidden dimension
      num_layers: stacked LSTM layers
      dropout: dropout between layers and before head
      forecast_steps: number of future timesteps to predict (k)
      output_size: number of output features per timestep (e.g., 1)
    """

    def __init__(
        self,
        model_param: LSTMModelParam,
    ):
        super().__init__()
        self.hidden_size = model_param.hidden_size
        self.num_layers = model_param.num_layers
        self.forecast_steps = model_param.forecast_steps
        self.output_size = model_param.output_size

        self.lstm = nn.LSTM(
            input_size=model_param.input_size,
            hidden_size=model_param.hidden_size,
            num_layers=model_param.num_layers,
            batch_first=True,
            dropout=model_param.dropout if model_param.num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(model_param.hidden_size)
        self.dropout = nn.Dropout(model_param.dropout)

        # head: map last hidden state -> forecast_steps * output_size
        self.head = nn.Sequential(
            nn.Linear(model_param.hidden_size, model_param.hidden_size),
            nn.ReLU(),
            nn.Dropout(model_param.dropout),
            nn.Linear(
                model_param.hidden_size,
                model_param.forecast_steps * model_param.output_size,
            ),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                # forget gate bias (per LSTM param layout): set to 1
                param.data[n // 4 : n // 2].fill_(1.0)

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (batch, T_input, D)
        returns: (batch, forecast_steps, output_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, T_input, hidden)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)
        out = self.head(out)  # (batch, forecast_steps * output_size)
        out = out.view(x.size(0), self.forecast_steps, self.output_size)
        return out

    def predict(self, x):
        """Returns raw forecasts (regression outputs)."""
        return self.forward(x)
