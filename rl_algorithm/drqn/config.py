import torch

embedding_size = 64
rnn_hidden_dim = 64
rnd_out_dim = 16
rnn_layer_dim = 1
episode_batch_size = 1
max_episode_num = 5000
discount = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
