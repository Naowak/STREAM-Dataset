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



class DynamicGRU(nn.Module):
    def __init__(self, input_dim, output_dim, target_params):
        super(DynamicGRU, self).__init__()
        
        # 3*H^2 + (3*I + 6 + O)*H + (O - target_params) = 0
        a = 3
        b = 3 * input_dim + 6 + output_dim
        c = output_dim - target_params
        
        delta = b**2 - 4*a*c
        if delta < 0:
            hidden_size = 1
        else:
            hidden_size = int((-b + math.sqrt(delta)) / (2 * a))
            hidden_size = max(1, hidden_size)
            
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        gru_out, _ = self.gru(x)
        logits = self.fc(gru_out)
        return logits
    


class PositionalEncoding(nn.Module):
    """Encodage positionnel classique (Sin/Cos) pour donner la notion du temps au Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, Time, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class DynamicTransformerDecoderOnly(nn.Module):
    """
    Architecture GPT-like : Uniquement le décodeur avec masque causal.
    Prédit de gauche à droite, simule l'approche autoregressive.
    """
    def __init__(self, input_dim, output_dim, target_params, num_layers=2, nhead=2):
        super(DynamicTransformerDecoderOnly, self).__init__()
        
        # Résolution pour trouver d_model (D)
        # P ≈ 12 * L * D^2 + (I + O) * D
        a = 12 * num_layers
        b = input_dim + output_dim
        c = -target_params
        
        delta = b**2 - 4*a*c
        if delta < 0:
            d_model = nhead # Fallback minimal
        else:
            d_model = int((-b + math.sqrt(delta)) / (2 * a))
            
        # Arrondir d_model pour qu'il soit divisible par nhead
        d_model = max(nhead, d_model - (d_model % nhead))
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Un Decoder-Only s'implémente souvent via un Encoder avec masque causal en Pytorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)
        
        # Création du masque causal pour empêcher de "regarder dans le futur"
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Application du transformer
        out = self.transformer(x_emb, mask=causal_mask, is_causal=True)
        logits = self.fc_out(out)
        return logits


class DynamicTransformerEncoderDecoder(nn.Module):
    """
    Architecture originale (Attention is All You Need - 2017).
    """
    def __init__(self, input_dim, output_dim, target_params, num_layers=2, nhead=2):
        super(DynamicTransformerEncoderDecoder, self).__init__()
        
        # Résolution pour trouver d_model (D)
        # L_enc + L_dec ≈ 28 * L * D^2 + (I + O) * D
        a = 28 * num_layers
        b = 2 * input_dim + output_dim # On utilise input_dim pour projeter src et tgt
        c = -target_params
        
        delta = b**2 - 4*a*c
        if delta < 0:
            d_model = nhead
        else:
            d_model = int((-b + math.sqrt(delta)) / (2 * a))
            
        d_model = max(nhead, d_model - (d_model % nhead))
        
        self.src_emb = nn.Linear(input_dim, d_model)
        self.tgt_emb = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=4*d_model, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, output_dim)
        
        self.actual_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # Pour une tâche synchrone [B, T, I] -> [B, T, O] sans génération token par token,
        # on utilise X comme source ET cible (avec masque causal sur la cible).
        
        src = self.pos_encoder(self.src_emb(x))
        tgt = self.pos_encoder(self.tgt_emb(x))
        
        seq_len = x.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_is_causal=True)
        logits = self.fc_out(out)
        return logits