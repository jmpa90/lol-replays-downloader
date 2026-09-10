"""Modelo hurdle para la RIM: P(cotiza) x E[RIM | cotiza].

  * Clasificador (GBTClassifier): y_cls = 1{RIM_target > 0}
  * Regresor (GBTRegressor) sobre filas con RIM_target > 0:
        y_reg = log(RIM_target / rim_ref_safe)
    donde rim_ref_safe es la ultima RIM > 0 conocida (o el IMM si no hay historia).
    Modelar el ratio (y no el nivel) hace que el modelo aprenda "cambios" (reajuste,
    gratificacion, aguinaldo, cambio de empleador) y generalice entre tramos de renta.

Salidas por fila:
  prob_cotiza, rim_condicional (= rim_ref_safe * exp(pred), topada), rim_esperada
  (= prob * rim_condicional, util para agregados de recaudacion) y rim_proyectada
  (punto: rim_condicional si prob >= umbral, si no 0; util para reemplazar el dato
  individual).

En produccion se puede sustituir GBT de MLlib por xgboost.spark.SparkXGBRegressor /
SynapseML LightGBM manteniendo la misma interfaz (fit / transform).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import Config
from .features import FEATURES_CATEGORICAS, FEATURES_NUMERICAS


def _etapas_features() -> tuple[list, list[str]]:
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                for c in FEATURES_CATEGORICAS]
    cols = list(FEATURES_NUMERICAS) + [f"{c}_idx" for c in FEATURES_CATEGORICAS]
    assembler = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep")
    return indexers + [assembler], cols


@dataclass
class HurdleModel:
    clasificador: PipelineModel
    regresor: PipelineModel
    columnas: list[str]
    umbral: float = 0.5
    version: str = "rim-hurdle-gbt-v1"

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(cls, train: DataFrame, cfg: Config, umbral: float = 0.5) -> "HurdleModel":
        etapas_c, cols = _etapas_features()
        gbt_c = GBTClassifier(labelCol="y_cls", featuresCol="features", maxIter=cfg.gbt_max_iter,
                              maxDepth=cfg.gbt_max_depth, stepSize=cfg.gbt_step_size,
                              subsamplingRate=cfg.gbt_subsampling, maxBins=64, seed=cfg.seed)
        clasificador = Pipeline(stages=etapas_c + [gbt_c]).fit(train)

        etapas_r, _ = _etapas_features()
        gbt_r = GBTRegressor(labelCol="y_reg", featuresCol="features", maxIter=cfg.gbt_max_iter,
                             maxDepth=cfg.gbt_max_depth, stepSize=cfg.gbt_step_size,
                             subsamplingRate=cfg.gbt_subsampling, maxBins=64, seed=cfg.seed, lossType="absolute")
        regresor = Pipeline(stages=etapas_r + [gbt_r]).fit(train.filter(F.col("y_cls") == 1.0))
        return cls(clasificador=clasificador, regresor=regresor, columnas=cols, umbral=umbral,
                   version=cfg.version_modelo)

    # ---------------------------------------------------------- transform
    def transform(self, df: DataFrame) -> DataFrame:
        prob = F.element_at(F.col("probability"), 2)   # vector -> P(y=1)
        out = (self.clasificador.transform(df)
               .withColumn("prob_cotiza", prob)
               .drop("features", "rawPrediction", "probability", "prediction",
                     *[f"{c}_idx" for c in FEATURES_CATEGORICAS]))
        out = (self.regresor.transform(out)
               .withColumnRenamed("prediction", "pred_log_ratio")
               .drop("features", *[f"{c}_idx" for c in FEATURES_CATEGORICAS]))
        return (out
                .withColumn("rim_condicional",
                            F.round(F.least(F.col("rim_ref_safe") * F.exp(F.col("pred_log_ratio")),
                                            F.col("tope_target")), 0))
                .withColumn("rim_esperada", F.round(F.col("prob_cotiza") * F.col("rim_condicional"), 0))
                .withColumn("rim_proyectada", F.when(F.col("prob_cotiza") >= self.umbral,
                                                     F.col("rim_condicional")).otherwise(0.0)))

    # ---------------------------------------------------------- persist
    def save(self, ruta: str) -> None:
        self.clasificador.write().overwrite().save(f"{ruta}/clasificador")
        self.regresor.write().overwrite().save(f"{ruta}/regresor")

    @classmethod
    def load(cls, ruta: str, cfg: Config, umbral: float = 0.5) -> "HurdleModel":
        _, cols = _etapas_features()
        return cls(clasificador=PipelineModel.load(f"{ruta}/clasificador"),
                   regresor=PipelineModel.load(f"{ruta}/regresor"), columnas=cols, umbral=umbral,
                   version=cfg.version_modelo)

    # ---------------------------------------------------- importancias
    def importancias(self, top: int = 25) -> pd.DataFrame:
        gc = self.clasificador.stages[-1].featureImportances.toArray()
        gr = self.regresor.stages[-1].featureImportances.toArray()
        df = pd.DataFrame({"feature": self.columnas, "imp_clasificador": gc, "imp_regresor": gr})
        df["imp_total"] = df["imp_clasificador"] + df["imp_regresor"]
        return df.sort_values("imp_total", ascending=False).head(top).reset_index(drop=True)
