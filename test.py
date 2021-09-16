import model
import numpy as np
import matplotlib.pyplot as plt


def main():
    X = np.load('sample_img.npy') # shape = (5,728)
    
    dnn = model.build_neural_net()
    
    print('dnn input shape:', X.shape)
    dnn_out = dnn.forward(X)
    print('dnn output shape', dnn_out.shape)
    print('\n')

    cnn = model.build_cnn()

    X = X.reshape(5,28,28,1)
    print('cnn input shape:', X.shape)
    
    cnn_out = cnn.forward(X)
    print('cnn output shape', cnn_out.shape)

    img = X[0].reshape(-1,28)

    plt.imshow(img)
    plt.title('sample img')
    plt.show()

if __name__ == '__main__':
    main()
