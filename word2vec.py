import os
from operator import add
import pandas as pd
import compress_pickle as pickle

from pyspark.sql import SparkSession, Window
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col, array_contains, split, udf, monotonically_increasing_id, lit, row_number, countDistinct
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, Normalizer, VectorAssembler, \
    CountVectorizer, Word2Vec, Word2VecModel
from pyspark.sql.types import *

from sklearn.metrics.pairwise import cosine_similarity as cosine

from nltk.stem.porter import *

conf=SparkConf()
conf.set("spark.driver.memory", "10g")
conf.set("spark.cores.max", "4")
conf.set("spark.executor.heartbeatInterval", "3600")

sc = SparkContext.getOrCreate(conf)

spark = SQLContext(sc)

df = spark.read.options(header="True").csv('data_nohtml.csv').dropna()
skills = spark.read.options(header="True").csv('skills.csv').dropna()

df.show()
skills.show()

tokenizer = Tokenizer(inputCol="FullDescription", outputCol="words")
wordsData = tokenizer.transform(df)
skills = tokenizer.transform(skills)

wordsData.select("Id", "words").show()
skills.select("words").show()

remover = StopWordsRemover(inputCol="words", outputCol="noStops")
stopwords = remover.getStopWords()
stopwords[:10]
wordsData = remover.transform(wordsData)
skills = remover.transform(skills)

wordsData.select("Id", "noStops").show()
skills.select("noStops").show()

stemmer = PorterStemmer()


def stem(word_list):
    output_word_list = []
    for word in word_list:
        word_stem = stemmer.stem(word)
        if len(word_stem) > 2:
            output_word_list.append(word_stem)
    return output_word_list


stemmer_udf = udf(lambda word: stem(word), ArrayType(StringType()))
wordsData = wordsData.withColumn("stemmedNoStops", stemmer_udf("noStops"))
skills = skills.withColumn("stemmedNoStops", stemmer_udf("noStops"))

wordsData.select("Id", "stemmedNoStops").show()
skills.select("stemmedNoStops").show()

# word2vec = Word2Vec(inputCol="stemmedNoStops", outputCol="vecs")
# model = word2vec.fit(wordsData)
# wordsData = model.transform(wordsData)
# model.save("vec.model")

# forbidden = "*.\"/\\[]:;|<>"
# sector_list = wordsData.toPandas()["Classification"].unique()
#
# vectors = []
# for sector in sector_list:
#     print(f"Doing sector {sector}")
#
#     if os.path.isfile(f"vectors.lz4"):
#         continue
#     rows = wordsData.select("vecs").count()
#     counts = wordsData.select("vecs").filter(f"Classification='{sector}'").rdd \
#         .map(lambda row: row['vecs'].toArray()) \
#         .reduce(lambda x,y: x+y)
#     counts = counts / rows
#     vectors.append((sector, counts))
# pickle.dump(vectors, f"vectors.lz4")

vectors = pickle.load("vectors.lz4")
new_model = Word2VecModel.load("vec.model")

cv = [("""
Experience in management
Law
Client Relations
Teamwork
""".replace("\n", " "),)]

schema = StructType([StructField("FullDescription", StringType(), True)])

df = spark.createDataFrame(data=cv, schema=schema)
df = tokenizer.transform(df)
df = remover.transform(df)
df = df.withColumn("stemmedNoStops", stemmer_udf("noStops"))
df = new_model.transform(df)
cv_vec = df.rdd.map(lambda row: row['vecs'].toArray()).reduce(lambda x,y: x+y)

sim = []
for vec in vectors:
    sim.append((vec[0], cosine(vec[1].reshape(1, -1), cv_vec.reshape(1, -1))))

new_sim = []
for item in sim:
    new_sim.append((item[0], item[1][0][0]))
sim = new_sim

sim.sort(key=lambda x: x[1], reverse=True)
for sector in sim:
    print(f"{sector[0]}: {sector[1]}")

