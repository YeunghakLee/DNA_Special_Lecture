import cv2
import numpy as np
import matplotlib.pyplot as plt

def makeTraindata():
    traindata = np.random.randint(0,100,(25,2)).astype(np.float32)
    resp = np.random.randint(0,2,(25,1)).astype(np.float32)
    return traindata, resp

def knn():
    traindata, resp = makeTraindata()

    red = traindata[resp.ravel()==0]
    blue = traindata[resp.ravel()==1]
    plt.scatter(red[:,0],red[:,1],80,'r','^')
    plt.scatter(blue[:,0],blue[:,1],80,'b','s')

    newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
    plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
    plt.show()

    knn = cv2.ml.KNearest_create()
    knn.train(traindata,cv2.ml.ROW_SAMPLE,resp)
    ret, results, neighbours, dist = knn.findNearest(newcomer,3)

    print(results, neighbours)

    return

knn()
    
