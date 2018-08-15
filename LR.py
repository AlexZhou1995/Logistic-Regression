import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

class LR(object):
    def __init__(self,seed=0):
        self.w = None
        self.log = []
        np.random.seed(seed)

    def reset(self,seed=0):
        self.w = None
        np.random.seed(seed)
        self.log = []

    def GD_fit(self, x, y, n_iters,lr=1e-2):
        m,d = x.shape
        w = np.random.random([1,d+1])
        x = np.hstack([x,np.ones([m,1])])

        for i in range(n_iters):
            loss = self._logloss(x, y, w)
            print('loss={:.4f}'.format(loss))
            self.log.append(loss)

            grad = self._grad(x,y,w)
            w -= lr * grad

        self.w = w

    def SGD(self,x,y,n_iters,batch_size,lr=1e-2):
        m, d = x.shape
        w = np.random.random([1, d + 1])
        x = np.hstack([x, np.ones([m, 1])])

        for i in range(n_iters):
            loss = self._logloss(x, y, w)
            print('loss={:.4f}'.format(loss))
            self.log.append(loss)

            idx = np.arange(m)
            np.random.shuffle(idx)
            n_batches = m//batch_size
            for j in range(n_batches):
                grad = self._grad(x[idx[j*batch_size:(j+1)*batch_size],:], y[idx[j*batch_size:(j+1)*batch_size]], w)
                w -= lr * grad
            if m % batch_size != 0:
                grad = self._grad(x[idx[n_batches * batch_size:], :],
                                  y[idx[n_batches * batch_size:]], w)
                w -= lr * grad

        self.w = w

    def SAG_fit(self,x,y,n_iters,lr=1e-2):
        m, d = x.shape
        w = np.random.random([1, d + 1])
        x = np.hstack([x, np.ones([m, 1])])

        d = np.zeros(w.shape)
        gs = np.zeros([m,w.shape[1]])
        for i in range(n_iters):
            loss = self._logloss(x, y, w)
            print('loss={:.4f}'.format(loss))
            self.log.append(loss)

            idx = np.random.randint(m)
            d = d - gs[idx,:] + self._grad(x[idx,:],y[idx],w)
            gs[idx,:] = self._grad(x[idx,:],y[idx],w)
            w -= lr*d/m

        self.w = w

    def SVRG_fit(self,x,y,n_iters,freq,lr=1e-2):
        m, d = x.shape
        w = np.random.random([1, d + 1])
        x = np.hstack([x, np.ones([m, 1])])

        w0 = np.random.random([1, d + 1])
        for i in range(n_iters):
            loss = self._logloss(x, y, w)
            print('loss={:.4f}'.format(loss))
            self.log.append(loss)

            w = w0
            mu = self._grad(x,y,w0)
            w0 = w
            for t in range(freq):
                idx = np.random.randint(m)
                w0 = w0 - lr * (self._grad(x[idx,:],y[idx],w0) - self._grad(x[idx,:],y[idx],w) + mu)

        self.w = w



    def predict(self,x,w=None):
        m, d = x.shape
        x = np.hstack([x, np.ones([m,1])])
        w = self.w if w is None else w

        score = self._logistic(x,w)
        y = deepcopy(score)
        y[y>=0.5]=1
        y[y<0.5]=0
        return y,score

    def _logistic(self,x,w):
        return np.squeeze(1.0/(1.0+np.exp(-np.matmul(w,x.T))))

    def _logloss(self,x,y,w):
        y_ = self._logistic(x,w)
        loss = -1*np.mean(y.T*np.log(y_) + (1-y.T)*np.log(1-y_))
        return loss

    def _grad(self,x,y,w):
        if len(y)>1:
            grad = np.matmul((self._logistic(x,w) - y.T), x)/x.shape[0]
        else:
            grad = (self._logistic(x,w) - y)*x
        return grad



Xp = np.random.multivariate_normal([1,-1,1,-1,2,1,5,5],2*np.ones([8,8]),1000)
Xn = np.random.multivariate_normal([-1,1,-1,1,1,2,5,6],2*np.ones([8,8]),1000)

X = np.vstack([Xp,Xn])
Y = np.vstack([np.ones([1000,1]), np.zeros([1000,1])])

lrc = LR()

lrc.GD_fit(X,Y,10)
log1 = lrc.log

lrc.reset()
lrc.SGD(X,Y,10,100)
log2 = lrc.log

lrc.reset()
lrc.SAG_fit(X,Y,2000)
log3 = lrc.log

lrc.reset()
lrc.SVRG_fit(X,Y,100,100)
log4 = lrc.log

print(log1[-1],log2[-1],log3[-1],log4[-1])
