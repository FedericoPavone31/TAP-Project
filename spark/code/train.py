from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, IDF, HashingTF
from pyspark.ml.classification import LogisticRegression


spark = SparkSession.builder \
    .appName("Spark Sentiment Analysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "12g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.default.parallelism", "44") \
    .getOrCreate()

schema="CommentID string, VideoID string, VideoTitle string, AuthorName string, CommentText string, Sentiment string, PublishedAt string, label short"

dataset = (
    spark.read.format("csv")
    .schema(schema)
    .option("escape", "\"")
    .load(
        "/home/federico/YTSentiment/spark/dataset/dataset.csv.gz"
    )
)

tokenizer = Tokenizer(inputCol="CommentText", outputCol="words")
hashingtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)
LRegression = LogisticRegression(featuresCol= 'features', labelCol= 'label',maxIter=100)
TFIDFPipeline = Pipeline(stages=[tokenizer, hashingtf, idf, LRegression])

TFIDFModel=TFIDFPipeline.fit(dataset)

TFIDFModelSummary = TFIDFModel.stages[-1].summary
print(f"Model accuracy: {TFIDFModelSummary.accuracy}")

TFIDFModel.write().overwrite().save("/home/federico/YTSentiment/spark/code/models/YTSentiment_model")
print("Model saved")

spark.stop()
