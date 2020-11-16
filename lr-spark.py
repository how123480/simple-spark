# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path


from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark import SparkConf, SparkContext


def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf)
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """
    feats = line.strip().split(",")
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1]
    feats = feats[: len(feats) - 1]
    #feats.insert(0,label)
    features = [ float(feature) for feature in feats ] # need floats
    return LabeledPoint(label, np.array(features))

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("/cccs_hw4/data_banknote_authentication.txt")
parsedData = data.map(mapper)

# Train model
model = LogisticRegressionWithSGD.train(parsedData)
# Predict the first elem will be actual data and the second
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (point.label, model.predict(point.features)))
# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda r: r[0] != r[1]).count() / float(parsedData.count())
# Print some stuff
print("Training Error = " + str(trainErr))