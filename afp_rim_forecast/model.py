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

import json
import os
from dataclasses import dataclass, field

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
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


UMBRALES_GRILLA = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


@dataclass
class HurdleModel:
    clasificador: PipelineModel
    regresor: PipelineModel
    columnas: list[str]
    umbrales: dict[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.5})   # por horizonte
    smearing: dict[int, float] = field(default_factory=lambda: {1: 1.0, 2: 1.0})   # por horizonte
    version: str = "rim-hurdle-gbt-v1"

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(cls, train: DataFrame, cfg: Config, calibrar: bool = True) -> "HurdleModel":
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
        modelo = cls(clasificador=clasificador, regresor=regresor, columnas=cols, version=cfg.version_modelo)
        if calibrar:
            modelo.calibrar(train)
        return modelo

    # ---------------------------------------------------------- calibrar
    def calibrar(self, df: DataFrame) -> pd.DataFrame:
        """Elige por horizonte el umbral que minimiza el WAPE de la prediccion puntual y el
        factor de "smearing" s_h = sum(real) / sum(p * rim_cond) que corrige el sesgo de
        rim_esperada (el regresor de log-ratio con perdida absoluta estima una mediana).
        Se calibra sobre datos etiquetados (entrenamiento o validacion), nunca sobre test."""
        base = self._puntaje(df).select("horizonte", "rim_real", "prob_cotiza", "rim_condicional")
        aggs = [F.sum("rim_real").alias("_real"),
                F.sum(F.col("prob_cotiza") * F.col("rim_condicional")).alias("_esp")]
        for u in UMBRALES_GRILLA:
            pred = F.when(F.col("prob_cotiza") >= u, F.col("rim_condicional")).otherwise(0.0)
            aggs.append((F.sum(F.abs(pred - F.col("rim_real"))) / F.sum(F.abs("rim_real"))).alias(f"w_{u:.2f}"))
        res = base.groupBy("horizonte").agg(*aggs).toPandas()
        filas = []
        for _, r in res.iterrows():
            h = int(r["horizonte"])
            wapes = {u: r[f"w_{u:.2f}"] for u in UMBRALES_GRILLA}
            self.umbrales[h] = min(wapes, key=wapes.get)
            self.smearing[h] = float(r["_real"] / r["_esp"]) if r["_esp"] else 1.0
            filas.append({"horizonte": h, "umbral": self.umbrales[h], "smearing": self.smearing[h],
                          **{f"wape@{u:.2f}": w for u, w in wapes.items()}})
        return pd.DataFrame(filas)

    def _umbral_col(self):
        expr = F.lit(0.5)
        for h, u in self.umbrales.items():
            expr = F.when(F.col("horizonte") == h, F.lit(u)).otherwise(expr)
        return expr

    def _smearing_col(self):
        expr = F.lit(1.0)
        for h, sm in self.smearing.items():
            expr = F.when(F.col("horizonte") == h, F.lit(sm)).otherwise(expr)
        return expr

    # ---------------------------------------------------------- transform
    def _puntaje(self, df: DataFrame) -> DataFrame:
        prob = vector_to_array(F.col("probability"))[1]   # vector -> P(y=1)
        out = (self.clasificador.transform(df)
               .withColumn("prob_cotiza", prob)
               .drop("features", "rawPrediction", "probability", "prediction",
                     *[f"{c}_idx" for c in FEATURES_CATEGORICAS]))
        out = (self.regresor.transform(out)
               .withColumnRenamed("prediction", "pred_log_ratio")
               .drop("features", *[f"{c}_idx" for c in FEATURES_CATEGORICAS]))
        return out.withColumn("rim_condicional",
                              F.round(F.least(F.col("rim_ref_safe") * F.exp(F.col("pred_log_ratio")),
                                              F.col("tope_target")), 0))

    def transform(self, df: DataFrame) -> DataFrame:
        return (self._puntaje(df)
                .withColumn("rim_esperada",
                            F.round(F.least(F.col("prob_cotiza") * F.col("rim_condicional") * self._smearing_col(),
                                            F.col("tope_target")), 0))
                .withColumn("rim_proyectada", F.when(F.col("prob_cotiza") >= self._umbral_col(),
                                                     F.col("rim_condicional")).otherwise(0.0)))

    # ---------------------------------------------------------- persist
    def save(self, ruta: str) -> None:
        self.clasificador.write().overwrite().save(f"{ruta}/clasificador")
        self.regresor.write().overwrite().save(f"{ruta}/regresor")
        os.makedirs(ruta, exist_ok=True)
        with open(f"{ruta}/calibracion.json", "w") as fh:
            json.dump({"umbrales": self.umbrales, "smearing": self.smearing, "version": self.version}, fh)

    @classmethod
    def load(cls, ruta: str, cfg: Config) -> "HurdleModel":
        _, cols = _etapas_features()
        with open(f"{ruta}/calibracion.json") as fh:
            cal = json.load(fh)
        return cls(clasificador=PipelineModel.load(f"{ruta}/clasificador"),
                   regresor=PipelineModel.load(f"{ruta}/regresor"), columnas=cols,
                   umbrales={int(k): v for k, v in cal["umbrales"].items()},
                   smearing={int(k): v for k, v in cal["smearing"].items()},
                   version=cal.get("version", cfg.version_modelo))

    # ---------------------------------------------------- importancias
    def importancias(self, top: int = 25) -> pd.DataFrame:
        gc = self.clasificador.stages[-1].featureImportances.toArray()
        gr = self.regresor.stages[-1].featureImportances.toArray()
        df = pd.DataFrame({"feature": self.columnas, "imp_clasificador": gc, "imp_regresor": gr})
        df["imp_total"] = df["imp_clasificador"] + df["imp_regresor"]
        return df.sort_values("imp_total", ascending=False).head(top).reset_index(drop=True)
