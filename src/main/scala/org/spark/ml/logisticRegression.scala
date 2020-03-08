package org.spark.ml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.log4j._

object logisticRegression extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

  val data = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load("/Users/ankurpareek/Projects/SparkML/src/main/resources/Data/HousingPrice.csv")


  data.printSchema()

}
