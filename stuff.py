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

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, Normalizer, VectorAssembler, CountVectorizer
from pyspark.sql.types import *

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

# hashingTF = HashingTF(inputCol="stemmedNoStops", outputCol="rawFeatures", numFeatures=200)
# wordsData = hashingTF.transform(wordsData)
# skills = hashingTF.transform(skills)
#
# wordsData.select("Id", "rawFeatures").show()
# skills.select("words", "rawFeatures").show()

cv = CountVectorizer(inputCol="stemmedNoStops", outputCol="counts")
model = cv.fit(wordsData)
wordsData = model.transform(wordsData)

wordsData.select("Id", "counts").show()

# idf = IDF(inputCol="rawFeatures", outputCol="features")
# idfModel = idf.fit(wordsData)
# wordsData = idfModel.transform(wordsData)
# skills = idfModel.transform(skills)
#
# wordsData.select("Id", "features").show()
# skills.select("words", "features").show()
#
#
# normalizer = Normalizer(inputCol="features", outputCol="norm")
# wordsData = normalizer.transform(wordsData)
# skills = normalizer.transform(skills)
#
# wordsData.select("Id", "norm").show()
# skills.select("words", "norm").show()

skillsNoStops = skills.select("stemmedNoStops").collect()
skillset_list = []
for row in skillsNoStops:
    for item in row[0]:
        skillset_list.append(item)
skillset = set(skillset_list)

vocab = sc.parallelize(model.vocabulary)

def parser(item):
    if item[0] in skillset and item[1] > 0.0:
        return [item]
    else:
        return []

forbidden = "*.\"/\\[]:;|<>"
sector_list = wordsData.toPandas()["SubClassification"].unique()
for sector in sector_list:
    print(f"Doing sector {sector}")
    filename = ''.join(c for c in sector if c not in forbidden)
    if os.path.isfile(f"skills_{filename}.lz4"):
        continue
    counts = wordsData.select("counts").filter(f"SubClassification='{sector}'").rdd \
        .map(lambda row: row['counts'].toArray()) \
        .reduce(lambda x,y: [x[i]+y[i] for i in range(len(y))])
    total_counts = sorted(list(zip(model.vocabulary, counts)), key=lambda x: x[1], reverse=True)
    total_counts = sc.parallelize(total_counts).flatMap(parser).collect()

    pickle.dump((sector, pd.DataFrame(total_counts, columns=["Words", "Frequency"])), f"skills_{filename}.lz4")


# list(zip(model.vocabulary, counts[i]['counts'].values))


# skillset = set(skills.select("FullDescription").collect())
# jobs = []
# def inSkillSet(row):
#     for i in range(len(row.words)):
#         if row.words[i] in skillset:
#             row.freqList.append(row.norm[i])
#     return row
#
# wordsData = wordsData.rdd.map(inSkillSet)
#
# print(jobs)

# jobsWithID = wordsData.withColumn("monot_id", monotonically_increasing_id())
# skillsWithID = skills.withColumn("monot_id", monotonically_increasing_id())
#
# window = Window.orderBy(col('monot_id'))
# jobsWithID = jobsWithID.withColumn("id", row_number().over(window))
# skillsWithID = skillsWithID.withColumn("id", row_number().over(window))
#
# jobsMat = IndexedRowMatrix(jobsWithID.select("id", "norm").rdd.map(lambda row: IndexedRow(row.id, row.norm.toArray()))).toBlockMatrix()
# skillsMat = IndexedRowMatrix(skillsWithID.select("id", "norm").rdd.map(lambda row: IndexedRow(row.id, row.norm.toArray()))).toBlockMatrix()
#
# print(str(jobsMat.numRows()) + " " + str(jobsMat.numCols()))
# print(str(skillsMat.numRows()) + " " + str(skillsMat.numCols()))
#
# dot = jobsMat.multiply(skillsMat.transpose())
# dotIndex = dot.toIndexedRowMatrix().rows
# print(dotIndex.first())
#
# def findHighestSkill(row):
#     sort = sorted(enumerate(row.vector), key=lambda x: x[1], reverse=True)
#     print(str(row.index) + " " + str(sort[0]))
#     return
#
# dotIndex.foreach(findHighestSkill)
#
# jobsWithID.filter("id=2117").show()
# skillsWithID.filter("id=19").show()