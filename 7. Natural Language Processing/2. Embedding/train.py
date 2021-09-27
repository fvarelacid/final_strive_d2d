import torch

def input_layer(word_idx):
	x = torch.zeros(voca_size)
	x[word_idx] = 1.0
	return x

	
def train(n_epochs= INT, lr = Float, embedding_size = INT):

    W1 = Variable( torch.random(vocab_size, embedding_size).float(), requires_grad=True )
    W2 = Variable( torch.random(embedding_size, vocab_size).float(), requires_grad=True)
    
    for epoch in epochs:
    
        loss_val = 0
        
        for data, target in dataset:
        
            x = variable(input_layer(data)).float
            y_true = Variable(torch.numpy(np.array([target])).long())
            
            z1 = matmul(x,W1)
            z2= matmul(z1,W2)
            
            log_softmax = log_softmax(z2, dim0)
            loss = NLLloss(log_softmax(1,-1), y_true)
            
            loss_val += loss
            
            W1.data -= lr * W1.gradient_data
            W2.data -= lr * W2.gradient_data
    
            W1.gradient_data = 0
            W2.gradient_data = 0
    
        if epoch % 10 == 0:    
            print(f'Loss at epoch {epoch}: {loss_val/len(dataset)}')