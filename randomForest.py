from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
import os
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'



sc = SparkContext(appName="randomForest")
# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, '/liyuanshuo/randomForestData.data')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainClassifier(trainingData, numClasses=8, categoricalFeaturesInfo={},
                                     numTrees=20, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=18, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
actual = labelsAndPredictions.filter(lambda v_p: v_p[0] == v_p[1]).count() / float(testData.count())

print('***'*10 + "打印随机森林的相关信息" + '***' * 10)
print('Random Forest precision: ' + str(actual))
print('Learned classification forest model:')
print(model.toDebugString())
print('***'*10 +  '***' * 10)

# Save and load model
# model.save(sc, "myModelPath")
# sameModel = RandomForestModel.load(sc, "myModelPath")
