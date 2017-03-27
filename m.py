import numpy as np
import sys
from mnist import MNIST
from collections import defaultdict

STEP_SIZE = 0.01
REGU = 0.000001

class ReLuLayer(object):
  def __init__(self, N,M):
    #self.w=np.random.randn(N,M)
    self.w=np.random.randn(N,M)
    self.w/=np.sqrt(N/2)
    self.b=np.zeros((1,M))
  def fwd(self, x):  
    #print(x)
    self.y=np.maximum(0, np.dot(x, self.w) + self.b)
    self.yy = (self.y>0)
    self.x=x
    return self.y
  def backprop(self, ds):
    dx = (ds*self.yy).dot(self.w.T)
    dw = np.dot(self.x.T,(ds*self.yy))
    db = np.sum(ds, axis=0, keepdims=True)
    dw += REGU * self.w  # L2 regularization
    self.w-=dw * STEP_SIZE
    self.b-=db * STEP_SIZE
    return dx

class OutLayer(object):
  def __init__(self, N,M):
    self.w=np.random.randn(N,M)
    self.w/=np.sqrt(N)
    self.b=np.zeros((1,M))
  def fwd(self, x):  
    #print(x)
    self.y=np.dot(x, self.w) + self.b
    self.x=x
    return self.y
  def backprop(self, ds):
    dx = np.dot(ds, self.w.T)
    dw = np.dot(self.x.T, ds)
    db = np.sum(ds, axis=0, keepdims=True)
    dw += REGU * self.w  # L2 regularization
    self.w -= dw * STEP_SIZE
    self.b -= db * STEP_SIZE
    return dx

def train(max_epoch,mndata):
  l1=ReLuLayer(28*28,300)
  l2=ReLuLayer(300,100)
  l3=ReLuLayer(100,10)
  lo=OutLayer(10,10)
  net=[l1,l2, l3,lo]
  
  train_img, train_label = mndata.load_training()

  for ii in range(max_epoch):
    i=np.random.randint(len(train_img))
    x=np.array([train_img[i]])/255.0
    assert(x.shape==(1,28*28))
    y=train_label[i]
    #print(x,y)
    c=x
    for l in net:
      c=l.fwd(c)
    if np.isnan(c).any():
      raise Exception("nan weights in network")
    exp_scores = np.exp(c)
    probs = exp_scores/np.sum(exp_scores,axis=1, keepdims=True)
    #print (x,c,probs)
    loss = -np.log(probs[0,y])
    for l in net:
      loss += np.sum(l.w**2)*REGU
    if ii%10==0:
      print("loss",loss)
  
    ds = probs
    ds[0,y] -= 1
    #print(ds)
    for l in net[::-1]:
      ds=l.backprop(ds)

  return net
      
def test(net,mndata):
  test_img, test_label = mndata.load_testing()

  g=defaultdict(lambda:defaultdict(int))
  for x,y in zip(test_img,test_label):
    c=np.array([x])/255.0
    for l in net:
      c=l.fwd(c)
    exp_scores = np.exp(c)
    probs = exp_scores/np.sum(exp_scores,axis=1, keepdims=True)
    loss = -np.log(probs[0,y])
    yg=np.argmax(probs, axis=1)[0]
    g[y][yg]+=1
  
  print (g)
  for k,v in sorted(g.items()):
    print("true class %s: %s/%s"%(k,v[k],sum(v.values())))
    

def run():
  max_epoch=int(sys.argv[1])
  mndata = MNIST('./mnist')
  net = train(max_epoch,mndata)
  import pickle
  with open("net.pkl","wb") as f:
    pickle.dump(net,f,protocol=2)
  test(net,mndata)


if __name__ == "__main__":
  run()

