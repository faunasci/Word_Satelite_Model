import torch
import torch.nn as nn
import torch.nn.functional as F

class SatelliteEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, levels=3):
        """
        Args:
            vocab_size: Tamanho do vocabulário.
            d_model: Dimensão máxima (para o Nível 0).
            levels: Número de níveis de hierarquia.
        """
        super().__init__()
        self.d_model = d_model
        self.levels = levels
        
        # Definindo dimensões para cada nível
        # Nível 0: 100% d_model
        # Nível 1: 50% d_model
        # Nível 2: 25% d_model
        self.dims = [d_model, d_model // 2, d_model // 4]
        
        # Criando embeddings separados para cada "conceito" de dimensionalidade
        # Na prática, para simplificar o PoC, vamos manter um embedding único mas mascarar a entrada
        # Ou melhor: Ter embeddings reais menores e projetar.
        
        self.emb_l0 = nn.Embedding(vocab_size, self.dims[0])
        self.emb_l1 = nn.Embedding(vocab_size, self.dims[1])
        self.emb_l2 = nn.Embedding(vocab_size, self.dims[2])
        
        # Projetores para trazer de volta à dimensão do modelo (para o Transformer processar)
        # Isso simula a "reconstrução" do sinal no bus principal, mas a informação original era comprimida.
        self.proj_l1 = nn.Linear(self.dims[1], d_model)
        self.proj_l2 = nn.Linear(self.dims[2], d_model)

    def forward(self, x, levels):
        """
        x: [batch, seq_len] (indices dos tokens)
        levels: [batch, seq_len] (nível hierárquico de cada token: 0, 1 ou 2)
        
        Versão otimizada para memória: processa apenas os tokens necessários para cada nível
        """
        batch_size, seq_len = x.shape
        out = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        
        # Processamento Nível 0 (Full Tensor)
        mask_l0 = (levels == 0)
        if mask_l0.any():
            # Pegar apenas os índices que são nível 0
            x_l0 = torch.where(mask_l0, x, torch.zeros_like(x))
            emb0 = self.emb_l0(x_l0)
            out = torch.where(mask_l0.unsqueeze(-1), emb0, out)
            del emb0, x_l0  # Liberar memória
            
        # Processamento Nível 1 (Medium Tensor)
        mask_l1 = (levels == 1)
        if mask_l1.any():
            x_l1 = torch.where(mask_l1, x, torch.zeros_like(x))
            emb1 = self.emb_l1(x_l1)
            emb1_proj = self.proj_l1(emb1)
            out = torch.where(mask_l1.unsqueeze(-1), emb1_proj, out)
            del emb1, emb1_proj, x_l1
            
        # Processamento Nível 2 (Low Tensor)
        mask_l2 = (levels == 2)
        if mask_l2.any():
            x_l2 = torch.where(mask_l2, x, torch.zeros_like(x))
            emb2 = self.emb_l2(x_l2)
            emb2_proj = self.proj_l2(emb2)
            out = torch.where(mask_l2.unsqueeze(-1), emb2_proj, out)
            del emb2, emb2_proj, x_l2
            
        return out

class SatelliteLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, n_layer=4, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = SatelliteEmbedding(vocab_size, d_model)
        
        # Transformer Decoder padrão com Dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Cabeça de saída
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Pré-criar máscara causal (reutilizável)
        self.register_buffer('causal_mask', 
                            torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1))
        
    def forward(self, x, levels, mask=None):
        # x: [batch, seq_len]
        # levels: [batch, seq_len]
        
        # 1. Embedding Hierárquico
        x_emb = self.embedding(x, levels)
        
        # 2. Transformer
        # Usar a máscara causal pré-criada, cortada para o tamanho correto
        if mask is None:
            seq_len = x.size(1)
            mask = self.causal_mask[:seq_len, :seq_len]
            
        out = self.transformer(x_emb, mask=mask)
        
        # 3. Predição
        logits = self.fc_out(out)
        return logits
