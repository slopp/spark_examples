from pyspark import SparkContext
from pyspark.sql import *
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline

sc = SparkContext(appName="create_rf")
sqlContext = SQLContext(sc)

df = sqlContext.read.parquet("/home/sean/Other/spark_examples/iris-parquet")

# first, translate Species from string to index
stringIndexer = StringIndexer(inputCol="Species", outputCol="indexed").fit(df)
td = stringIndexer.transform(df)

# build a vector from our features
featureIndexer =VectorAssembler(inputCols=["Petal_Length"], outputCol="features")
tdd = featureIndexer.transform(td)

#build classifier and fit (create model)
dt = RandomForestClassifier(maxDepth=2, labelCol="indexed", featuresCol = "features")
model = dt.fit(tdd)
print(model)

model.save("/home/sean/Other/spark_examples/pyspark_rf")

# Test that we can load the model back into python
same_model = RandomForestClassificationModel.load("/home/sean/Other/spark_examples/pyspark_rf")
print(same_model)