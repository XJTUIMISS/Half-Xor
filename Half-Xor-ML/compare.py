import torch
import pickle
import torch.optim as optim
import time
import numpy as np
import os
import math
import multiprocessing
import random
from ctypes import *
from functools import partial
from random import randrange
import shutil
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool, Queue
import struct
import logging, traceback

from mlp import netdir
from network import Model, testModel
from dataset_gene import m


comUpper = int(pow(10, 9.01))
comLower = 1
queue = Queue()


def file2array(path, delimiter=' '):
    fp = open(path, 'r', encoding='utf-8')
    string = fp.read()              # read all
    fp.close()
    row_list = string.splitlines()  # Default Parameters '\n'
    data_list = [[int(i) for i in row.strip().split(delimiter)] for row in row_list]
    return np.array(data_list)

def plotFunction(filename, x, y, x2 = None, y2 = None, x3 = None, y3 = None):

    fig = plt.figure(num = 1,dpi = 120)
    ax = plt.subplot(111)
    isNone_inf = lambda x:[float("inf"), float("inf")] if x is None else x
    isNone_zero = lambda x:[0.0, 0.0] if x is None else x

    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.plot(x, y, label = "Multilayer perceptron", color ="blueviolet",\
             linestyle='-', marker='o', markersize = 2)
    if x2 is not None and y2 is not None:
        ax.plot(x2,y2,label = "Original(EZ)",color ="red",\
                 linestyle='-', marker='*', markersize = 3)

    ax.set_xlabel("cardinality(power10) m="+str(m),color='black')
    ax.set_ylabel("RMSD",color='black')
    plt.legend()
    plt.xlim(comLower - 0.5, math.log10(comUpper) + 0.5)
    plt.ylim(0, 0.01 + max([max(y), max(isNone_zero(y2)), max(isNone_zero(y3))]) )
    plt.show()
    plt.savefig(os.path.join("drawres", str(filename)))


def calCar(eleNum):
    try:
        magni = int(np.log10(eleNum))
        array = np.load(f"./data/half_ratio/{m}-{magni}0.npy")

        X = array[:, 0 : 34 - 1 - 1]
        y = array[:, 34 - 1]
        V = array[:, 34 - 1 - 1]


        sketch_cal, net_cal = 0, 0
        sketch_rmsd, net_rmsd = 0, 0
        for i in range(len(y)):
            nRes = model(torch.tensor(X[i], dtype = torch.float32)).item()
            nRes = nRes  if nRes > 3 * m else - m * math.log(V[i])
            net_cal += nRes
            net_rmsd += pow((nRes - eleNum) / eleNum, 2)
            print(y[i], model(torch.tensor(X[i], dtype = torch.float32)).item(), nRes, (nRes - eleNum) / eleNum)
        print("Count", eleNum, len(y), '\n')
        queue.put(struct.pack('iffff', eleNum, sketch_cal / len(y), \
            sketch_rmsd / len(y), net_cal / len(y), net_rmsd / len(y)))
    except Exception as e:
        logging.warning(traceback.print_exc())

if __name__ == "__main__":

    model = torch.load(os.path.join(netdir,  'net.pt'))
    print(testModel(model))
    cpu_num = multiprocessing.cpu_count()
    
    train_dir = "./Intermediate"
    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    
    pool = multiprocessing.Pool(cpu_num)
    sketchRes, netRes, actual = [], [], []
    myCardi = MyNumbers()
    proNow, proTotal = 0, 0
    print(cpu_num)
    for i in iter(myCardi):

        eleNum = i
        pool.apply_async(calCar, args=(eleNum, ))
        print("Creat a subprocess with Cardinality and number:", eleNum, f'{proNow} / {proTotal} ')
        proNow += 1
        proTotal += 1
        
        try:
            if proNow < int(cpu_num / 3 ): assert False
            elif proNow < 25: package = queue.get(timeout=60) # s
            else: 
                print("Pool full.")
                package = queue.get() # s

            proNow -= 1
            eleNum, sketch_cal, sketch_rmsd, net_cal, net_rmsd = struct.unpack('iffff', package)
        
            actual.append(eleNum)
            sketchRes.append(math.sqrt(sketch_rmsd ))
            netRes.append(math.sqrt(net_rmsd ))
                
            print("total:", eleNum)
            print("Xor_Estimate:", sketch_cal , "RMSD:", math.sqrt(sketch_rmsd ))
            print("Perceptron:", net_cal , "RMSD:", math.sqrt(net_rmsd))
            print('\n')
        except:
            continue
    

    pool.close()
    pool.join()
    print("Subprocesses finished.")
    recieveNum = 0
    while not len(actual) == proTotal:
        package = queue.get() # s
        eleNum, sketch_cal, sketch_rmsd, net_cal, net_rmsd = struct.unpack('iffff', package)
        recieveNum += 1
        print("Cardinadity", eleNum, "finished.")
        
        actual.append(eleNum)
        sketchRes.append(math.sqrt(sketch_rmsd ))
        netRes.append(math.sqrt(net_rmsd ))
                
        print("total:", eleNum)
        print("Xor_Estimate:", sketch_cal , "RMSD:", math.sqrt(sketch_rmsd ))
        print("Perceptron:", net_cal , "RMSD:", math.sqrt(net_rmsd ))
        print('\n')
        

    
    print("Data generation completed.")
    
    x1 = np.log10(actual)
    y1 = np.array(sketchRes)
    arrIndex1 = np.array(x1).argsort()
    y1 = y1[arrIndex1]
    x1 = x1[arrIndex1]

    x2 = np.log10(actual)
    y2 = np.array(netRes)
    arrIndex2 = np.array(x2).argsort()
    y2 = y2[arrIndex2]
    x2 = x2[arrIndex2]
    madrange = np.array([i for i in iter(MyNumbers())])
    plotFunction('m='+str(m)+'.png', x2, y2)

    actual = [i for i in x1]
    netRes = [i for i in y2]
    sketchRes =  [i for i in y1]

    print("actually", np.shape(actual), np.power(10, actual))
    print("sketchRes", np.shape(sketchRes), sketchRes)
    print("netRes", np.shape(netRes), netRes)

    
    