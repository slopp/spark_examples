from pyspark import SparkContext
from pyspark.sql import *
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml import PipelineModel

sc = SparkContext(appName="test")
sqlContext = SQLContext(sc)

model = RandomForestClassificationModel.load("/home/sean/Other/spark_examples/sparklyr_rf")

print(model)
