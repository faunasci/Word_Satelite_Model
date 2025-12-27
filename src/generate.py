import torch
import pickle
from src.model import SatelliteLLM
from src.data_processor import TextProcessor

def load_model(model_path='satellite_llm_model.pt', vocab_path='vocabulary.pkl', device=None):
    """Carrega o modelo treinado."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recriar modelo
    model = SatelliteLLM(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Carregar vocabulário
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    return model, vocab_data, device

def generate_text(prompt, model, vocab, idx_to_word, device, max_length=50, temperature=0.8):
    """Gera texto a partir de um prompt."""
    # Criar um processador temporário para encoding
    processor = TextProcessor()
    processor.vocab = vocab
    processor.idx_to_word = idx_to_word
    processor.nlp = None  # Vamos usar fallback simples
    
    # Encode prompt
    tokens, levels = processor._process_line(prompt)
    
    # Converter para tensors
    curr_tokens = torch.tensor([tokens]).to(device)
    curr_levels = torch.tensor([levels]).to(device)
    
    generated_tokens = tokens.copy()
    
    print(f"Prompt: {prompt}")
    print(f"Gerando...\n")
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(curr_tokens, curr_levels)
            
            # Pegar logits do último token
            last_logits = logits[0, -1, :] / temperature
            
            # Softmax para probabilidades
            probs = torch.softmax(last_logits, dim=0)
            
            # Sample do próximo token
            next_token = torch.multinomial(probs, 1).item()
            
            # Parar se gerar EOS
            if next_token == vocab.get("<EOS>", -1):
                break
            
            generated_tokens.append(next_token)
            
            # Atualizar tensores (assumir nível 2 para novos tokens)
            next_token_tensor = torch.tensor([[next_token]]).to(device)
            next_level_tensor = torch.tensor([[2]]).to(device)
            
            curr_tokens = torch.cat([curr_tokens, next_token_tensor], dim=1)
            curr_levels = torch.cat([curr_levels, next_level_tensor], dim=1)
            
            # Limitar tamanho do contexto para não estourar memória
            if curr_tokens.size(1) > 100:
                curr_tokens = curr_tokens[:, -100:]
                curr_levels = curr_levels[:, -100:]
    
    # Decodificar
    result = " ".join([idx_to_word.get(t, "<UNK>") for t in generated_tokens])
    return result

if __name__ == "__main__":
    print("=== Satellite LLM - Text Generation ===\n")
    
    # Carregar modelo
    print("Carregando modelo...")
    model, vocab_data, device = load_model()
    print(f"✓ Modelo carregado (device: {device})\n")
    
    # Prompts de teste
    prompts = [
        "The artificial intelligence",
        "In the future",
        "Language models are",
        "Chronicles games",
        "The satellite model"
    ]
    
    print("--- Gerações de Teste ---\n")
    for prompt in prompts:
        result = generate_text(
            prompt, 
            model, 
            vocab_data['vocab'], 
            vocab_data['idx_to_word'], 
            device,
            max_length=30,
            temperature=0.2
        )
        print(f"OUTPUT: {result}\n")
        print("-" * 80)
