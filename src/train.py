import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

# Ajuste para encontrar o módulo 'src' quando rodando de dentro da pasta src no Colab
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if '/content' not in sys.path:
    sys.path.append('/content')

from src.model import SatelliteLLM
from src.data_processor import TextProcessor

# Configurações (ajustadas para GPU de 4GB)
BATCH_SIZE = 16  # Reduzido de 32
SEQ_LEN = 35
D_MODEL = 64     # Reduzido de 128
N_HEAD = 4
N_LAYER = 2      # Reduzido de 4
EPOCHS = 8
LR = 0.001
MAX_VOCAB = 10000

def get_batch(source_tokens, source_levels, i, device):
    """
    source_tokens: [batch_size, total_seq_len]
    source_levels: [batch_size, total_seq_len]
    i: índice de início na dimensão de sequência
    """
    seq_len = min(SEQ_LEN, source_tokens.size(1) - 1 - i)
    # Pegar fatias da dimensão de sequência (dim=1), não da dimensão de batch (dim=0)
    data = source_tokens[:, i:i+seq_len].to(device)
    target = source_tokens[:, i+1:i+1+seq_len].to(device)
    levels = source_levels[:, i:i+seq_len].to(device)
    return data, target, levels

def make_batches(tokens, levels):
    # Cortar o resto que não cabe no batch size
    n_batch = len(tokens) // BATCH_SIZE
    tokens = tokens[:n_batch * BATCH_SIZE]
    levels = levels[:n_batch * BATCH_SIZE]
    
    # Reshape para [batch_size, -1]
    # Isso permite processar chunks contínuos
    tokens = tokens.view(BATCH_SIZE, -1)
    levels = levels.view(BATCH_SIZE, -1)
    return tokens, levels

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Preparar Dados
    processor = TextProcessor(max_vocab_size=MAX_VOCAB)
    
    print("Carregando dados...")
    processor.build_vocab_from_file("data/wikitext-2-train.txt")
    
    # Carregar e processar (limitando linhas para teste rápido se necessário, mas vamos tentar full)
    train_tokens, train_levels = processor.encode_file("data/wikitext-2-train.txt")
    val_tokens, val_levels = processor.encode_file("data/wikitext-2-valid.txt")
    
    # Preparar Batches (Manter na CPU por enquanto)
    train_data, train_lvl_data = make_batches(train_tokens, train_levels)
    val_data, val_lvl_data = make_batches(val_tokens, val_levels)
    
    # train_data = train_data.to(device)  <-- REMOVIDO
    # ...
    
    vocab_size = len(processor.vocab)
    print(f"Vocabulário final: {vocab_size}")
    
    # 2. Inicializar ou Carregar Modelo
    import os
    model_path = 'satellite_llm_model.pt'
    
    model = SatelliteLLM(vocab_size=vocab_size, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER).to(device)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists(model_path):
        print("\n🔄 Modelo existente encontrado! Carregando para continuar treinamento...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_loss = checkpoint.get('final_val_loss', float('inf'))
            start_epoch = checkpoint.get('epochs_trained', 0)
            print(f"✓ Modelo carregado (já treinou {start_epoch} épocas, best val loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"⚠️  Erro ao carregar modelo: {e}")
            print("→ Começando treinamento do zero...")
            start_epoch = 0
    else:
        print("\n🆕 Criando novo modelo do zero...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Loop de Treino
    print(f"\nIniciando treinamento por {EPOCHS} épocas...")
    if start_epoch > 0:
        print(f"(Continuando da época {start_epoch + 1})")
    
    for epoch in range(EPOCHS):
        actual_epoch = start_epoch + epoch + 1
        model.train()
        total_loss = 0.
        start_time = time.time()
        
        # Iterar sobre a dimensão de sequência
        num_batches = train_data.size(1) // SEQ_LEN
        
        for batch, i in enumerate(range(0, train_data.size(1) - 1, SEQ_LEN)):
            data, targets, levels = get_batch(train_data, train_lvl_data, i, device)
            
            optimizer.zero_grad()
            output = model(data, levels)
            loss = criterion(output.view(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch % 200 == 0 and batch > 0:
                cur_loss = total_loss / 200
                elapsed = time.time() - start_time
                print(f'| Época {actual_epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                      f'lr {LR} | ms/batch {elapsed*1000/200:5.2f} | '
                      f'loss {cur_loss:5.2f}')
                total_loss = 0
                start_time = time.time()
                
        # Validação
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for i in range(0, val_data.size(1) - 1, SEQ_LEN):
                data, targets, levels = get_batch(val_data, val_lvl_data, i, device)
                output = model(data, levels)
                loss = criterion(output.view(-1, vocab_size), targets.reshape(-1))
                val_loss += loss.item()
        
        val_loss /= (val_data.size(1) // SEQ_LEN)
        print(f'-' * 89)
        print(f'| Fim da época {actual_epoch:3d} | valid loss {val_loss:5.2f}', end='')
        
        # Indicar se é o melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(' ← MELHOR! ⭐')
        else:
            print()
        print(f'-' * 89)

    # 4. Análise de Eficiência Final (Baseada no dataset de validação)
    print("\n--- Análise de Eficiência (Dataset Validação) ---")
    flat_levels = val_levels.view(-1)
    unique, counts = torch.unique(flat_levels, return_counts=True)
    level_counts = dict(zip(unique.tolist(), counts.tolist()))
    
    total_tokens = sum(level_counts.values())
    
    cost_standard = total_tokens * D_MODEL
    cost_satellite = (level_counts.get(0, 0) * D_MODEL) + \
                     (level_counts.get(1, 0) * (D_MODEL // 2)) + \
                     (level_counts.get(2, 0) * (D_MODEL // 4))
                     
    savings = 100 * (1 - cost_satellite / cost_standard)
    
    print(f"Total Tokens Validação: {total_tokens}")
    print(f"Distribuição:")
    print(f"  - Nível 0 (Sujeitos): {level_counts.get(0, 0)} ({level_counts.get(0,0)/total_tokens*100:.1f}%)")
    print(f"  - Nível 1 (Verbos)  : {level_counts.get(1, 0)} ({level_counts.get(1,0)/total_tokens*100:.1f}%)")
    print(f"  - Nível 2 (Outros)  : {level_counts.get(2, 0)} ({level_counts.get(2,0)/total_tokens*100:.1f}%)")
    print(f"ECONOMIA TEÓRICA DE MEMÓRIA: {savings:.2f}%")
    
    # 5. Salvar Modelo
    print("\n--- Salvando Modelo ---")
    import pickle
    
    total_epochs = start_epoch + EPOCHS
    
    # Salvar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'final_val_loss': best_val_loss,
        'epochs_trained': total_epochs,
    }, 'satellite_llm_model.pt')
    
    # Salvar vocabulário
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump({
            'vocab': processor.vocab,
            'idx_to_word': processor.idx_to_word
        }, f)
    
    print(f"✓ Modelo salvo em: satellite_llm_model.pt")
    print(f"✓ Vocabulário salvo em: vocabulary.pkl")
    print(f"✓ Total de épocas treinadas: {total_epochs}")
    print(f"✓ Melhor val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
