class ModelParam:
    dropout: float
    output_size: int
    forecast_steps: int


class LSTMModelParam(ModelParam):
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int


class TCNModelParam(ModelParam):
    in_channels: int
    out_channels: int
    kernel_size: int
    num_channels: list[int]
