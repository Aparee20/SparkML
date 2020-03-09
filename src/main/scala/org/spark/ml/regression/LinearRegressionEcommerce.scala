package org.spark.ml.regression

import org.apache.log4j._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object LinearRegressionEcommerce extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

  val data = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load("/Users/ankurpareek/Projects/SparkML/src/main/resources/Data/Ecommerce.csv")


  data.printSchema()
  data.head(1)

  data.columns.foreach(println)

  // (label,features)


  val df =
    data.select(
      data("Yearly Amount Spent").as("label"),
      data("Time on App"),
      data("Avg Session Length"),
      data("Time on Website"),
      data("Length of Membership")
    )


  df.printSchema()


  //setting up all the feature  to vector into 1  columns

  import org.apache.spark.ml.feature.VectorAssembler


  val assembler = new VectorAssembler()
    .setInputCols(
      Array(
        "Time on App",
        "Avg Session Length",
        "Time on Website",
        "Length of Membership"
        )).setOutputCol("features")


  val output = assembler.transform(df).select("label", "features")
  output.show(10)


  val lr = new LinearRegression()

  val lrModel = lr.fit(output)

  val traningSummary = lrModel.summary

  traningSummary.residuals.show()

   println(
     "RMSE " +traningSummary.rootMeanSquaredError
     + ",\n"
     + "SE "+ traningSummary.meanSquaredError
     + ",\n"
     + "R2 "+ traningSummary.r2)




}
