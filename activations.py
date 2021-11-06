import numpy as np
from layers import Layer, LossLayer
import warnings

class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.act = None 
    
    def forward(self,X):
        self.act = np.maximum(0,X)

        return self.act 
    
    def backward(self, grad_act):

        grad = grad_act.copy()
        grad[self.act <= 0] = 0 
        return grad
    

class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.act = None 
    
    def forward(self, X):
        X_shift = X - X.max(axis = -1, keepdims = True)
        norm = np.exp(X_shift)
        denom = norm.sum(axis=-1, keepdims = True)
        self.act = norm/denom
        
        return self.act
    
    def backward(self, grad_act):
        n, d = self.act.shape 
        idt_mtx = np.identity(d)

        def calc_single_grad(act, prev_grad):
            act = act[None]
            prev_grad = prev_grad[None]
            d_soft = act * idt_mtx - act.T@act
            
            return prev_grad@d_soft
        
        grads = np.zeros(self.act.shape)
        for i, (act, grad) in enumerate(zip(self.act, grad_act)):
            grads[i] = calc_single_grad(act, grad)
        

        return grads
        # derivative = output - y, grad_act already is that
        # return grad_act


class CrossEntropy(LossLayer):
    def __init__(self):
        super().__init__()
        self.act = None
        self.inp = None 
        self.eps = 1e-9
    
    def forward(self, pred, y):
        pred = np.clip(pred, self.eps, 1. - self.eps)
        self.act = -np.sum(y*np.log(pred + self.eps))/len(pred)

        self.inp = (pred, y)
        
        return self.act 


    def backward(self, grad_act = None):
        pred, y = self.inp 
        return -y/pred * (1/len(pred))


def test_cross_grad():
    softmax = Softmax()
    criterion = CrossEntropy()

    n, c = 100, 10
    X = np.random.rand(n, c)

    y_idxs = np.random.choice(10, n)
    y = np.zeros((n,c))
    y[list(range(n)), y_idxs] = 1.


    pred = softmax(X)
    _ = criterion(X, y)
    c_grad = criterion.backward()
    s_grad = softmax.backward(c_grad)

    print((s_grad - (pred - y)).sum())


if __name__ == "__main__":
    test_cross_grad()
