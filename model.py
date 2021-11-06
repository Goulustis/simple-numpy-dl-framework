import numpy as np
from tqdm import tqdm

from layers import Fc, Conv2d
from activations import Relu, Softmax, CrossEntropy

class Sequential:

    def __init__(self, layers = [], lr = 0.1, classes = 10, training = True, epochs = 10, batch_size = 500, loss_fnc = CrossEntropy):
        self.layers = layers 
        self.optim = GD(lr, self.layers)
        self.training = training
        self.epochs = epochs
        self.batch_size = batch_size
        self.classes = classes
        self.loss_fnc = loss_fnc()

    
    def fit(self, X, y):
        epochs = self.epochs
        batch_size = self.batch_size
        n = X.shape[0]

        self.training = True
        # for i in tqdm(range(epochs)):
        for i in range(epochs):
            print('---------- epoch %s of %s----------------'%(i, epochs))
            X_batches, y_batches = gen_batch(X,y, self.batch_size)

            epoch_loss = 0
            # for X_batch, y_batch in zip(X_batches,y_batches):
            for X_batch, y_batch in tqdm(zip(X_batches,y_batches), total = len(X_batches)):
                pred = self.forward(X_batch)
                loss = self.loss_fnc(pred, y_batch)
                self.optim.step()

                epoch_loss += np.abs(loss).sum()/batch_size

                # print('curr_loss:', (first_grad**2).sum())
            
            print('loss: %s'%(epoch_loss))
    

    
    def predict(self, X):
        self.training = False
        return self.forward(X)
        


    def forward(self, X):
        n = X.shape[0]
        act = X 
        for l in self.layers:
            act = l.forward(act)
        
        if self.training == False:
            act = act.reshape(n,self.classes)
            act = np.argmax(act, axis = -1)
        
        return act
    
    def backward(self):
        g = self.loss_fnc.backward() 

        for l in reversed(self.layers):
            g = l.backward(g)


# optimizer
class GD:
    def __init__(self, lr, params):
        self.lr = lr 
        self.params = params
    
    def step(self):

        for l in self.params:
            ws, g = l.get_weights(), l.get_grads()
        
            if None is ws[0]:
                continue
        
            (w,b) = ws 
            (gradw, gradb) = g
            w_new = w - self.lr*gradw
            b_new = b - self.lr*gradb
            l.update_weights(w_new, b_new)
    
    def zero_grad(self):
        for l in self.params:
            l.zero_grad()

# helper function, maybe make another utils?
def gen_batch(X,y, batch_size = 100):
    # m, d = X.shape
    m = X.shape[0]
    indexes = np.random.permutation(m)

    X_batches = []
    y_batches = []
    for i in range(m//batch_size - 1):
        X_batches.append(X[indexes[batch_size * i : batch_size*(i + 1)]])
        y_batches.append(y[indexes[batch_size * i : batch_size*(i + 1)]])
    
    if m%batch_size != 0:
        X_batches.append(X[indexes[-(m%batch_size):]])
        y_batches.append(y[indexes[-(m%batch_size):]])

    return X_batches, y_batches

def build_cnn(epochs = 10, batch_size = 512):
    conv1 = Conv2d(1,16,3,2,name = 1)
    conv2 = Conv2d(16,32,3,2,name = 2)
    conv3 = Conv2d(32,64,3,2,name = 3) 
    conv4 = Conv2d(64,10,2,1, name = 4)

    layers = [conv1, Relu(), conv2, Relu(), conv3, Relu(), conv4, Softmax()]

    network = Sequential(layers, epochs = epochs, batch_size=batch_size)

    return network

def build_neural_net(epochs = 10, batch_size = 512):

    fc1 = Fc(28**2,128)
    fc2 = Fc(128,64)
    fc3 = Fc(64,32)
    fc4 = Fc(32,10)

    layers = [fc1, Relu(),fc2,Relu(),fc3,Relu(),fc4,Softmax()]

    network = Sequential(layers, epochs = epochs, batch_size=batch_size)

    return network

