import torch

embedding_size = 64
word_embedding_size = 32
text_embedding_size = 128
batch_size = 256
discount = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
