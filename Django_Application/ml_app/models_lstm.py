# ml_app/models_lstm.py
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size=None,
        hidden_size=None,
        num_layers=None,
        num_classes=None,
        dropout=0.0,
        bidirectional=False,
    ):

        super().__init__()

        # Coerce None -> sane defaults
        input_size = (
            input_size if isinstance(input_size, int) and input_size > 0 else 2048
        )
        hidden_size = (
            hidden_size if isinstance(hidden_size, int) and hidden_size > 0 else 512
        )
        num_layers = num_layers if isinstance(num_layers, int) and num_layers > 0 else 2
        num_classes = (
            num_classes if isinstance(num_classes, int) and num_classes > 0 else 2
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x_1_T_D):
        out, _ = self.lstm(x_1_T_D)
        last = out[:, -1, :]
        return self.fc(last)
