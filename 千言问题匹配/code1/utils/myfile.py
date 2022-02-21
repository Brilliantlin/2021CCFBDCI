import json
import os
import pickle
def removeFile(path):
    '''递归删除文件夹下所有'''
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def makedirs(prefix):
    '''prefix：文件夹目录，可以递归生成'''
    if not os.path.exists(prefix):
        os.makedirs(prefix)

def readFile(filename):
    '''

    Args:
        filename:

    with open(filename,'r',encoding='utf-8') as f:
        s = f.readlines()
    return s


    '''
    with open(filename,'r',encoding='utf-8') as f:
        s = f.readlines()
    return s
def loadjson(filename):
    '''

    :param filename: 需要加载的json 文件路径
    :return: json内容
    '''
    with open(filename,'r') as f:
        contents = json.load(f)
    return contents
def savejson(obj,filename):
    '''

    :param obj:json 变量
    :param filename: 存储文件名
    :return: None
    '''
    with open(filename,'w') as f:
        json.dump(obj,f,indent=1)

def savePkl(config, filepath):
    f = open(filepath, 'wb')
    pickle.dump(config, f)
    f.close()

def loadPkl(filepath):
    f = open(filepath, 'rb')
    config = pickle.load(f)
    return config