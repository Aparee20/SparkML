package org.spark.ml.regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.log4j._

object logisticRegressionHousing extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

  val data = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load("/Users/ankurpareek/Projects/SparkML/src/main/resources/Data/HousingPrice.csv")


  data.printSchema()

  data.head(1)


  // (label,features)

  import org.apache.spark.ml.feature.VectorAssembler
  import org.apache.spark.ml.linalg.Vectors


  val df =
    data.select(
      data("Price").as("label"),
      data("Avg Area House Age"),
      data("Avg Area Income"),
      data("Avg Area Number of Rooms"),
      data("Avg Area Number of Bedrooms"),
      data("Area Population")
    )


  df.printSchema()


  //setting up all the feature  to vector into 1  columns

  import org.apache.spark.ml.feature.VectorAssembler
  import org.apache.spark.ml.linalg.Vectors


  val assembler = new VectorAssembler()
    .setInputCols(
      Array("Avg Area House Age",
        "Avg Area Income",
        "Avg Area Number of Rooms",
        "Avg Area Number of Bedrooms",
        "Area Population")).setOutputCol("features")


  val output = assembler.transform(df).select("label", "features")
  output.show(10)


  val lr = new LinearRegression()

  val lrModel = lr.fit(output)

  val traningSummary = lrModel.summary

  traningSummary.residuals.show()


}
