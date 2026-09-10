"""Backtest con origenes moviles, baselines y metricas.

Metricas (por horizonte y por segmento):
  * MAE, RMSE, WAPE = sum|e| / sum|y|  (robusta a ceros, a diferencia del MAPE)
  * sesgo_agregado = (sum pred - sum real) / sum real   (error de recaudacion proyectada)
  * AUC / Brier del clasificador y exactitud cero/no-cero de la prediccion puntual
Baselines que el modelo debe superar:
  * persistencia:  RIM del ultimo mes conocido (rim_l1)
  * media_3:       promedio de los ultimos 3 meses conocidos
  * estacional:    RIM del mismo mes del anio anterior, reajustada por la variacion del IMM
"""
from __future__ import annotations

import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

PREDICTORES = ["rim_proyectada", "rim_esperada", "persistencia", "media_3", "estacional"]


def agregar_baselines(df: DataFrame) -> DataFrame:
    return (df.withColumn("persistencia", F.col("rim_l1"))
            .withColumn("estacional", F.when(F.col("rim_mismo_mes_ly") > 0,
                                             F.least(F.col("rim_mismo_mes_ly") * (1 + F.col("imm_var_12m_target")),
                                                     F.col("tope_target")))
                        .otherwise(F.col("rim_l1"))))


def metricas(df: DataFrame, por: list[str] | None = None) -> pd.DataFrame:
    """Tabla larga de metricas por predictor (y por columnas de corte opcionales)."""
    por = ["horizonte"] + (por or [])
    filas = []
    for p in PREDICTORES:
        e = F.col(p) - F.col("rim_real")
        agg = (df.groupBy(*por).agg(
            F.count("*").alias("n"),
            F.avg(F.abs(e)).alias("mae"),
            F.sqrt(F.avg(e * e)).alias("rmse"),
            (F.sum(F.abs(e)) / F.sum(F.abs(F.col("rim_real")))).alias("wape"),
            ((F.sum(F.col(p)) - F.sum("rim_real")) / F.sum("rim_real")).alias("sesgo_agregado"),
            F.avg(((F.col(p) > 0) == (F.col("rim_real") > 0)).cast("double")).alias("exactitud_cero"),
            F.avg(F.when(F.col("rim_real") > 0, F.abs(e) / F.col("rim_real"))).alias("mape_cond"),
        ).withColumn("predictor", F.lit(p)))
        filas.append(agg.toPandas())
    out = pd.concat(filas, ignore_index=True)
    cols = ["predictor"] + por + ["n", "mae", "rmse", "wape", "sesgo_agregado", "exactitud_cero", "mape_cond"]
    return out[cols].sort_values(por + ["predictor"]).reset_index(drop=True)


def metricas_clasificador(df: DataFrame) -> pd.DataFrame:
    ev = BinaryClassificationEvaluator(labelCol="y_cls", rawPredictionCol="prob_cotiza", metricName="areaUnderROC")
    filas = []
    for h in [r[0] for r in df.select("horizonte").distinct().collect()]:
        d = df.filter(F.col("horizonte") == h)
        brier = d.select(F.avg((F.col("prob_cotiza") - F.col("y_cls")) ** 2)).first()[0]
        filas.append({"horizonte": h, "auc": ev.evaluate(d), "brier": brier,
                      "tasa_cotiza_real": d.select(F.avg("y_cls")).first()[0],
                      "tasa_cotiza_pred": d.select(F.avg((F.col("prob_cotiza") >= 0.5).cast("double"))).first()[0]})
    return pd.DataFrame(filas).sort_values("horizonte").reset_index(drop=True)


def tramo_renta(col: str = "rim_ref_safe"):
    c = F.col(col)
    return (F.when(c <= 600_000, "1_<=600k").when(c <= 1_000_000, "2_600k-1M")
            .when(c <= 2_000_000, "3_1M-2M").otherwise("4_>2M"))
