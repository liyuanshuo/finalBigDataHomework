from pyspark import SparkContext
from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import os
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

def parseLine(line):
    parts = line.split(',')
    label = float(parts[0])
    features = Vectors.dense([float(x) for x in parts[1].split(' ')])
    return LabeledPoint(label, features)
sc = SparkContext(appName="Bayes")
data = sc.textFile('/liyuanshuo/data.data').map(parseLine)

# 将数据按照60%和40%的比例分为训练集和测试集
training, test = data.randomSplit([0.6, 0.4], seed=0)

# 训练朴素贝叶斯模型
model = NaiveBayes.train(training, 1.0)

# 预测准确率
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda x_v: x_v[0] == x_v[1]).count() / test.count()
print('***'*10 + "打印朴素贝叶斯的准确率：（0.6 / 0.4）" + '***' * 10)
print(accuracy)
print('***'*10 +  '***' * 10)