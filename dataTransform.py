from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
import numpy as np
import os
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'


def dataPreviousDealForBayes():
    conf = SparkConf().setAppName("dataTrandform")
    sc = SparkContext(conf=conf)
    textFile = sc.textFile("hdfs://liyuanshuo:9000/liyuanshuo/covtype.data")
    lines = textFile.collect()
    sc.stop()
    data = []
    for line in lines:
        # print('**' * 30)
        # print(line)
        # print('**' * 30)
        text = line.split(",")
        string = text[-1][-1]
        # print('**' * 30)
        # print(string)
        # print('**' * 30)
        data.append([int(string)])
        for i in range(len(text) - 1):
            data[-1].append(float(text[i]))


    data = np.array(data)
    for i in range(1, 11):
        data[:, i] = (data[:, i]-data[:, i].min())/(data[:, i].max()-data[:, i].min()+1)*3
    print('**' * 15 + "格式转换和数据预处理之后的数据" + '**' * 15 )
    print(data.shape)
    print(data)
    print('**' * 35 )

    fout = open('data.data', 'w')
    for i in range(len(data)):
        fout.write(str(int(data[i][0]))+",")
        for j in range(1, len(data[i])):
            fout.write(str(int(data[i][j]))+(j == len(data[i])-1 and "\n" or " "))
    fout.close()
    os.system("hadoop fs -put data.data /liyuanshuo")


def dataPreviousDealForDecisionTree():
    conf = SparkConf().setAppName("dataTrandform")
    sc = SparkContext(conf=conf)
    textFile = sc.textFile("hdfs://liyuanshuo:9000/liyuanshuo/covtype.data")
    lines = textFile.collect()
    sc.stop()
    data = []
    for line in lines:
        text = line.split(",")
        string = text[-1][-1]
        data.append([int(string)])
        for i in range(len(text) - 1):
            data[-1].append(int(text[i]))
    data = np.array(data)
    fout = open("decisionTreeData.data", "w")
    for i in range(len(data)):
        fout.write(str(int(data[i][0])))
        for j in range(1, len(data[i])):
            fout.write(" " + str(j) + ":" + str(data[i][j]))
        fout.write("\n")
    fout.close()
    print('**' * 15 + "格式转换和数据预处理之后的数据" + '**' * 15)
    print(data.shape)
    print(data)
    print('**' * 35)
    os.system("hadoop fs -put decisionTreeData.data /liyuanshuo")



def dataPreviousDealForRandomForest():
    conf = SparkConf().setAppName("dataTrandform")
    sc = SparkContext(conf=conf)
    textFile = sc.textFile("hdfs://liyuanshuo:9000/liyuanshuo/covtype.data")
    lines = textFile.collect()
    sc.stop()
    data = []
    for line in lines:
        text = line.split(",")
        string = text[-1][-1]
        data.append([int(string)])
        for i in range(len(text) - 1):
            data[-1].append(int(text[i]))
    data = np.array(data)
    fout = open("randomForestData.data", "w")
    for i in range(len(data)):
        fout.write(str(int(data[i][0])))
        for j in range(1, len(data[i])):
            fout.write(" " + str(j) + ":" + str(data[i][j]))
        fout.write("\n")
    fout.close()
    print('**' * 15 + "格式转换和数据预处理之后的数据" + '**' * 15)
    print(data.shape)
    print(data)
    print('**' * 35)
    os.system("hadoop fs -put randomForestData.data /liyuanshuo")

def dataPreviousDealForKnn():
    conf = SparkConf().setAppName("dataTrandform")
    sc = SparkContext(conf=conf)
    textFile = sc.textFile("hdfs://liyuanshuo:9000/liyuanshuo/covtype.data")
    lines = textFile.collect()
    sc.stop()
    sampleData = []
    label = []
    for line in lines:
        column = line.split(",")
        label.append(int(column[-1]))
        sampleData.append([])
        for i in range(len(column) - 1):
            sampleData[-1].append(float(column[i]))
    sampleData = np.array(sampleData)
    label = np.array(label)
    for i in range(0, 54):
        sampleData[:, i] = (sampleData[:, i] - sampleData[:, i].min()) / (
                    sampleData[:, i].max() - sampleData[:, i].min())
    fout = open("knnData.data", "w")
    for i in range(len(sampleData)):
        fout.write(str(i) + " " + str(label[i]))
        for j in range(len(sampleData[i])):
            fout.write(" " + str(sampleData[i][j]))
        fout.write("\n")
    fout.close()
    print('**' * 15 + "格式转换和数据预处理之后的数据" + '**' * 15)
    print(sampleData.shape)
    print(sampleData)
    print('**' * 35)
    os.system("hadoop fs -put knnData.data /liyuanshuo")


if __name__ == '__main__':
    # dataPreviousDealForBayes()
    # dataPreviousDealForDecisionTree()
    # dataPreviousDealForRandomForest()
    dataPreviousDealForKnn()