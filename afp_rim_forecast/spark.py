"""Creacion de la SparkSession (local para la simulacion, cluster en produccion)."""
from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "afp-rim-forecast", shuffle_partitions: int = 16,
              master: str | None = "local[*]", driver_memory: str = "4g") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", driver_memory)
    )
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
