import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple


class Word2Vec:
    def __init__(self, 
                 sequences: List[List[str]], 
                 embedding_dim: int = 100, 
                 window_size: int = 5,
                 negative_samples: int = 5):
        """
        Extremely efficient Word2Vec implementation.
        
        Args:
            sequences (List[List[str]]): Input sequences
            embedding_dim (int): Embedding dimension
            window_size (int): Context window size
            negative_samples (int): Number of negative samples
        """
        # Compute word frequencies
        self.word_freq = self._compute_word_frequencies(sequences)
        
        # Build vocabulary
        self.vocab = sorted(self.word_freq.keys(), key=lambda x: self.word_freq[x], reverse=True)
        self.vocab_size = len(self.vocab)
        
        # Create mappings
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        
        # Precompute training data
        self.training_data = self._prepare_training_data(sequences)
        
        # Initialize model
        self.model = SkipGramFast(self.vocab_size, embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def _compute_word_frequencies(self, sequences: List[List[str]]) -> Dict[str, int]:
        """
        Compute word frequencies efficiently.
        
        Args:
            sequences (List[List[str]]): Input sequences
        
        Returns:
            Dictionary of word frequencies
        """
        word_freq = {}
        for sequence in sequences:
            for word in sequence:
                word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq
    
    def _prepare_training_data(self, sequences: List[List[str]]) -> List[Tuple[int, int, int]]:
        """
        Prepare training data with efficient sampling.
        
        Args:
            sequences (List[List[str]]): Input sequences
        
        Returns:
            List of (center_word, context_word, label) tuples
        """
        training_data = []
        
        # Precompute negative sampling distribution
        word_freq_power = np.array([self.word_freq[word] ** 0.75 for word in self.vocab])
        word_freq_power /= word_freq_power.sum()
        
        for sequence in sequences:
            # Convert sequence to indices
            seq_indices = [self.word_to_index.get(word, -1) for word in sequence]
            seq_indices = [idx for idx in seq_indices if idx != -1]
            
            for i, center_idx in enumerate(seq_indices):
                # Define context window
                start = max(0, i - self.window_size)
                end = min(len(seq_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i == j:
                        continue
                    
                    context_idx = seq_indices[j]
                    
                    # Positive sample
                    training_data.append((center_idx, context_idx, 1))
                    
                    # Negative sampling
                    for _ in range(self.negative_samples):
                        # Sample negative word
                        negative_idx = np.random.choice(self.vocab_size, p=word_freq_power)
                        training_data.append((center_idx, negative_idx, 0))
        
        return training_data
    
    def train(self, epochs: int = 5, batch_size: int = 1024) -> float:
        """
        Train the Word2Vec model with batch processing.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        
        Returns:
            Final average loss
        """
        # Shuffle training data
        np.random.shuffle(self.training_data)
        
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Process in batches
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i+batch_size]
                
                # Prepare batch tensors
                center_words = torch.LongTensor([x[0] for x in batch])
                context_words = torch.LongTensor([x[1] for x in batch])
                labels = torch.FloatTensor([x[2] for x in batch])
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute loss
                loss = self.model(center_words, context_words, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Compute average loss
            avg_loss = epoch_loss / (len(self.training_data) // batch_size)
            total_loss += avg_loss
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return total_loss / epochs
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get embedding vector for a word.
        
        Args:
            word (str): Input word
        
        Returns:
            Numpy array of word embedding
        """
        if word not in self.word_to_index:
            return None
        
        idx = self.word_to_index[word]
        return self.model.input_embeddings.weight.data[idx].numpy()
    
    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar words.
        
        Args:
            word (str): Input word
            top_k (int): Number of similar words to return
        
        Returns:
            List of (word, similarity) tuples
        """
        if word not in self.word_to_index:
            return []
        
        # Get target word embedding
        target_embed = self.get_word_vector(word)
        
        # Compute similarities
        similarities = []
        for compare_word in self.vocab:
            if compare_word == word:
                continue
            
            # Get comparison embedding
            compare_embed = self.get_word_vector(compare_word)
            
            # Compute cosine similarity
            similarity = np.dot(target_embed, compare_embed) / (
                np.linalg.norm(target_embed) * np.linalg.norm(compare_embed)
            )
            similarities.append((compare_word, similarity))
        
        # Sort and return top k
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

class SkipGramFast(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Optimized Skip-Gram model with fast negative sampling.
        
        Args:
            vocab_size (int): Total number of unique words
            embedding_dim (int): Dimension of embeddings
        """
        super().__init__()
        
        # Input embeddings with efficient initialization
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.input_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        
        # Output embeddings with efficient initialization
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.output_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, center_words, context_words, labels):
        """
        Compute loss using efficient negative sampling.
        
        Args:
            center_words (torch.Tensor): Center word indices
            context_words (torch.Tensor): Context word indices
            labels (torch.Tensor): Positive/negative labels
        
        Returns:
            Computed loss
        """
        # Get embeddings
        input_embeds = self.input_embeddings(center_words)
        output_embeds = self.output_embeddings(context_words)
        
        # Compute dot product
        dot_product = torch.sum(input_embeds * output_embeds, dim=1)
        
        # Binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(dot_product, labels)
        
        return loss