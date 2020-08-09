# -*- coding: utf-8 -*-
from time import time
import logging,inspect
import pickle
from itertools import islice
import os.path

# dir_f = str(Path().resolve()) #+ "/lib/struc2vec/src"#os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
folder_pickles = os.path.abspath(os.getcwd()) + "/lib/struc2vec/pickles/" # + '/pickles/'

def returnPickle():
    return folder_pickles

def returnPathStruc2vec():
    return dir_f

def isPickle(fname):
    return os.path.isfile(folder_pickles+fname+'.pickle')

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def restoreVariableFromDisk(name):
    logging.info('Recovering variable...')
    t0 = time()
    val = None
    with open(folder_pickles + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
    t1 = time()
    logging.info('Variable recovered. Time: {}m'.format((t1-t0)/60))

    return val

def saveVariableOnDisk(f,name):
    logging.info('Saving variable on disk...')
    t0 = time()
    with open(folder_pickles + name + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()
    logging.info('Variable saved. Time: {}m'.format((t1-t0)/60))

    return





