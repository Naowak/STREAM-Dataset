# models.py
import math
import torch
import torch.nn as nn

class DynamicLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, target_params):
        super(DynamicLSTM, self).__init__()
        
        # Résolution de l'équation du second degré pour trouver hidden_size
        # 4*H^2 + (4*I + 4 + O)*H + (O - target_params) = 0
        a = 4
        b = 4 * input_dim + 4 + output_dim
        c = output_dim - target_params
        
        delta = b**2 - 4*a*c
        if delta < 0:
            hidden_size = 1 # Fallback de sécurité, bien que rare
        else:
            hidden_size = int((-b + math.sqrt(delta)) / (2 * a))
            hidden_size = max(1, hidden_size) # Au moins 1 de dimension
            
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits