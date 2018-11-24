from pyspark import SparkContext
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pyspark.mllib import tree
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'


sc = SparkContext(appName="DTTest")
# 这里对数据格式有严格要求，必须按照LabelPoint的RDD形式
data = MLUtils.loadLibSVMFile(sc, '/liyuanshuo/decisionTreeData.data')
# 这次将数据集进行划分，70%训练集， 30%测试集
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# 训练模型
# 空categoricalFeaturesInfo表示所有特征是连续的。代码直接参照官方的样例就可以，只不过部分参数需要自己设置一下
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  参数说明
#  numClasses:分类数，需比实际类别数量大，这里设置为8；
#  categoricalFeaturesInfo:特征类别信息，为空，意为所有特征为连续型变量；
#  impurity:信息纯度度量，进行分类时可选择熵或基尼，这里设置为基尼；
#  maxDepth:决策树最大深度，这里设为15；
#  maxBins:特征分裂时的最大划分数量,这里设为32。
model = DecisionTree.trainClassifier(trainingData, numClasses=8, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=15, maxBins=32)

# model = Pipeline([
#     ('ss', StandardScaler()),
#     ('DTC', DecisionTree.trainClassifier(trainingData, numClasses=8, categoricalFeaturesInfo={},
#                                      impurity='gini', maxDepth=15, maxBins=32))])

# model = model.fit(trainingData.map(lambda x: x.features))

# f = open('.\iris_tree.dot', 'w')
# tree.export_graphviz(model.get_params('DTC')['DTC'], out_file=f)

# tree.inherit_doc("decisionTree.dot")
# with open("decisionTree.dot", 'w') as f:
#     f = tree.export_graphviz(model, out_file=f)


# 预测准确率
# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
precision = labelsAndPredictions.filter(lambda v_p: v_p[0] == v_p[1]).count() / float(testData.count())
print('***'*10 + "打印决策树的准确率：（0.7 / 0.3）" + '***' * 10)
print('Decision Tree precision: ' + str(precision))
print('***'*10 +  '***' * 10)