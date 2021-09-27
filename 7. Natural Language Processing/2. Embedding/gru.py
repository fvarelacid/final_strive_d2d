'''from torch import nn

class CommandScorer(nn.Module):
	def __init__(self, input_size, hidden_size):
        	super(CommandScorer, self).__init__()
        	...
					self.embedding    = nn.Embedding(input_size, hidden_size)
					self.encoder_gru  = nn.GRU(hidden_size, hidden_size)
					...
	def forward(self, obs, commands, **kwargs):
					embedded = self.embedding(obs)
        	_, encoder_hidden = self.encoder_gru(embedded)
            ...
'''