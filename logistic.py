import sys
import os
import time
import math
SPARK_HOME = "/opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path


from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vectors
import numpy as np
from pyspark import SparkConf, SparkContext


class CustomLR():
	def __init__(self, learning_rate, iteration, sample_ratio):
		self.w = np.zeros((4,),dtype=np.float64)
		self.b = np.zeros((1,),dtype=np.float64)
		self.iter = iteration
		self.lr = learning_rate
		self.ratio = sample_ratio

	def _f(self,x):
		return 1/(1+np.exp(-(self.w.dot(x)+self.b)))
	def predict(self,x):
		#print("p: ", 1/(1+np.exp(-(np.dot(self.w,x)+self.b))))
		return 1 if self._f(x) > 0.5 else 0

	def gradient_w(self, data):
		return data.map(lambda x:((self._f(x.features)-x.label)*x.features)).reduce(lambda x,y:(x+y))

	def gradient_b(self, data):
		return data.map(lambda x:(self._f(x.features)-x.label)).reduce(lambda x,y:(x+y))

	def train(self,data):
		for i in range(self.iter):
			sub_data = data.sample(False,self.ratio)
			#print(type(self.gradient_w(sub_data)))
			self.w -= (self.lr)* self.gradient_w(sub_data)
			self.b -= (self.lr)* self.gradient_b(sub_data)
			if(i % 3 == 0):
				predict = data.map(lambda x:(x.label, self.predict(x.features)))
				error = predict.filter(lambda x:(x[0]!= x[1])).count() / data.count()
				print("iter: ",i)
				print("error: ", error)
				print("-"*30)

		return self.w, self.b

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

if __name__ == "__main__":
	start_time = time.time()
	sc = getSparkContext()
	print("configuration:")
	learning_rate = 0.01
	iteration = 30
	sample_ratio = 0.4
	print("learning rate: ", learning_rate)
	print("iteration: ", iteration)
	print("sample ratio: ", sample_ratio)
	print("-"*30)
	# Load and parse the data
	data = sc.textFile("/cccs_hw4/data_banknote_authentication.txt")
	parsedData = data.map(mapper)
	LR_model = CustomLR(learning_rate, \
		iteration, \
		sample_ratio)

	w,b = LR_model.train(parsedData)
	print("weight: ", w)
	print("bias: ", b)
	print("Time {}(s)".format(time.time() - start_time))
