from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
import os
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'


sc = SparkContext(appName="fileInfo")
spark = SparkSession.builder.getOrCreate()
filePath = "covtype.data"
df = spark.read.csv(filePath)
print('***' * 15 + "查看数据集的前两个" + '***' * 15)
print(df.head(2))
print('***' * 15 + "数据集的大小" + '***' * 15)
print(df.count())
print('***' * 15 + "数据集的描述" + '***' * 15)
print(df.describe())
print('***' * 35)