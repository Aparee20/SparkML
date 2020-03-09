package org.spark.ml.regression


//Sigmoid function (0/1)

//Accuracy
// TP +TN/total
// FP_FN/total


import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

object LogisticRegression extends  App
{


  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

  val data = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load("/Users/ankurpareek/Projects/SparkML/src/main/resources/Data/titanic.csv")


  data.printSchema()
  data.head(1)


  for (x <- data.columns)
    {
      println("data(\"" +x + "\"),")
    }


  val df =
    data.select(
      data("Survived").as("label"),
      data("Pclass"),
      data("Name"),
      data("Sex"),
      data("Age"),
      data("SibSp"),
      data("Parch"),
      data("Fare"),
      data("Embarked")
    )

  df.show()

  //drop missing values

  df.na.drop()

  df.show()



  //encode  one hot and string indexer


  import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}

  import org.apache.spark.ml.linalg.Vectors



  //converting string into numerical values
  val genderIndexer = (new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex"))
  val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")


  //converting numerical values into hot encoding

  val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
  val embarkedEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("Embarkedvec")


  val assembler = new VectorAssembler()
    .setInputCols(Array("Pclass","SexVec","Age","SibSp","Parch","Fare","Embarkedvec")).setOutputCol("features")

  //split data to train and test

  val Array(training,test) = df.randomSplit(Array(0.7,0.3),seed = 12345)


  import org.apache.spark.ml.Pipeline

  val lr = new LogisticRegression()

  val pipeline = new Pipeline().setStages(Array(genderIndexer,embarkedIndexer,genderEncoder,embarkedEncoder,assembler,lr))

  val model = pipeline.fit(training)

  val result = model.transform(test)






}
