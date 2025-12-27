import torch
import re
try:
    import spacy
except ImportError:
    spacy = None


import torch
import re
from collections import Counter
import os

try:
    import spacy
except ImportError:
    spacy = None

class TextProcessor:
    def __init__(self, max_vocab_size=10000, spacy_model="en_core_web_sm"):
        self.nlp = None
        if spacy:
            try:
                # Disable parser and ner for speed, we only need tagger
                self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
                print(f"Usando Spacy ({spacy_model}) para análise gramatical.")
            except OSError:
                print(f"Modelo Spacy '{spacy_model}' não encontrado. Tentando 'en_core_web_sm'...")
                try:
                    self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                    print("Usando Spacy (en_core_web_sm) como fallback.")
                except:
                    print("Nenhum modelo Spacy encontrado.")
            except Exception as e:
                print(f"Erro ao carregar Spacy: {e}")
        
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<EOS>"}
        self.next_id = 3
        self.max_vocab_size = max_vocab_size
        
    def build_vocab_from_file(self, filepath):
        print(f"Construindo vocabulário a partir de {filepath}...")
        counter = Counter()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Tokenização simples para contagem rápida
                words = re.findall(r'\w+|[^\w\s]', line.lower())
                counter.update(words)
        
        # Manter apenas os top N words
        most_common = counter.most_common(self.max_vocab_size - 3) # -3 para PAD, UNK, EOS
        
        for word, _ in most_common:
            self.vocab[word] = self.next_id
            self.idx_to_word[self.next_id] = word
            self.next_id += 1
            
        print(f"Vocabulário construído: {len(self.vocab)} tokens.")

    def encode_file(self, filepath, limit_lines=None):
        """
        Lê arquivo, processa POS tags e retorna tensores longos.
        """
        print(f"Processando arquivo {filepath}...")
        all_tokens = []
        all_levels = []
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if limit_lines and count >= limit_lines:
                    break
                
                tokens, levels = self._process_line(line)
                all_tokens.extend(tokens)
                all_levels.extend(levels)
                
                # Adicionar EOS
                all_tokens.append(self.vocab["<EOS>"])
                all_levels.append(2) # EOS como satélite (ou neutro)
                
                count += 1
                if count % 500 == 0:
                    print(f"Processadas {count} linhas...", end='\r')
                    
        print(f"\nTotal de tokens processados: {len(all_tokens)}")
        return torch.tensor(all_tokens), torch.tensor(all_levels)

    def _process_line(self, text):
        tokens = []
        levels = []
        
        if self.nlp:
            doc = self.nlp(text.lower())
            for token in doc:
                txt = token.text
                tid = self.vocab.get(txt, self.vocab["<UNK>"])
                tokens.append(tid)
                
                # Nível 0 (Sujeitos/Núcleos): NOUN, PROPN, PRON
                if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                    levels.append(0)
                # Nível 1 (Ação): VERB, AUX
                elif token.pos_ in ["VERB", "AUX"]:
                    levels.append(1)
                # Nível 2 (Satélites): ADJ, ADV, DET, PUNCT, etc.
                else:
                    levels.append(2)
        else:
            # Fallback
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for w in words:
                tid = self.vocab.get(w, self.vocab["<UNK>"])
                tokens.append(tid)
                if len(w) > 5: levels.append(0)
                elif len(w) > 3: levels.append(1)
                else: levels.append(2)
                
        return tokens, levels

    def decode(self, token_ids):
        return " ".join([self.idx_to_word.get(t.item(), "<UNK>") for t in token_ids])

