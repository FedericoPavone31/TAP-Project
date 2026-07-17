from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
import pyspark.sql.types as tp 
from pyspark.ml import PipelineModel

sparkConf = SparkConf().set("es.nodes", "elasticsearch").set("es.port", "9200")
spark = SparkSession.builder.appName("YoutubeSentiment").config(conf=sparkConf).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

kafkaServer="broker:9092"
topic = "yt-comments"
path = "/opt/sparkStream/models/YTSentiment_model"
elasticIndex = "yt-sentiment"

model = PipelineModel.load(path)
print("Model loaded")

schema = tp.StructType([
    tp.StructField(name="testo", dataType=tp.StringType(), nullable=True),
    tp.StructField(name="autore", dataType=tp.StringType(), nullable=True),
    tp.StructField(name="data", dataType=tp.StringType(), nullable=True),
    tp.StructField(name="id", dataType=tp.StringType(), nullable=True),
])

print("Reading from Kafka...")
df = spark.readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", kafkaServer) \
  .option("subscribe", topic) \
  .load()

df = df.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("parsedData")) \
    .select(
        col("parsedData.autore").alias("Autore"),
        col("parsedData.data").alias("Timestamp"),
        col("parsedData.testo").alias("CommentText"),
        col("parsedData.id").alias("CommentID")
    )

label_udf = udf(
    lambda x: {0: "Negativo", 1: "Neutro", 2: "Positivo"}.get(int(x), "None"),
    tp.StringType()
)

df = model.transform(df) \
    .withColumn("Sentiment", label_udf(col("prediction"))) \
    .select("Autore", "Timestamp", "CommentText", "CommentID", "Sentiment")

print("Writing to elastic...")
df.writeStream \
    .option("checkpointLocation", "/tmp/spark-checkpoints") \
    .format("es") \
    .start(elasticIndex) \
    .awaitTermination()
