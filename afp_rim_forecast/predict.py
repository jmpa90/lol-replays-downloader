"""Scoring de los meses abiertos: produce filas para la tabla `rim_proyectada`."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import Config
from .features import construir_features
from .model import HurdleModel

COLUMNAS_PROYECCION = [
    "afiliado_id", "periodo", "horizonte", "rim_valor", "origen", "detalle_origen",
    "prob_cotiza", "rim_condicional", "rim_esperada", "rim_predicha_previa",
    "n_pagadores_esperados", "tipo_afiliado",
    "periodo_snapshot", "periodo_reemplazo", "version_modelo", "fecha_calculo",
]


def proyectar(cfg: Config, tablas: dict[str, DataFrame], modelo: HurdleModel, periodo_snapshot: int,
              fecha_calculo: str, features: DataFrame | None = None) -> DataFrame:
    """Proyecta los `cfg.horizontes` meses posteriores al corte para todos los afiliados.

    Los afiliados cuyo mes target ya llego (empleador que paga anticipado) no se predicen:
    entran directamente como REAL / PAGO_ANTICIPADO.
    """
    spark = tablas["afiliados"].sparkSession
    if features is None:
        features = construir_features(spark, cfg, tablas, periodo_snapshot)
    # Un mes abierto esta "completo" si ya llegaron tantos pagadores como tenia el afiliado en el
    # ultimo mes conocido; si llego solo uno de varios empleadores, es un pago PARCIAL: se predice
    # el total y se toma al menos lo ya recibido.
    completo = (F.col("target_ya_conocido") == 1) & \
        (F.col("n_pagadores_ya_conocidos") >= F.greatest(F.col("n_pagadores_l1"), F.lit(1.0)))
    features = features.withColumn("_completo", completo.cast("int"))
    pred = modelo.transform(features.filter(F.col("_completo") == 0))
    parcial = F.col("target_ya_conocido") == 1
    predichas = (pred.select(
        "afiliado_id", F.col("periodo_target").alias("periodo"), F.col("horizonte").cast("int"),
        F.when(parcial, F.greatest(F.col("rim_proyectada"), F.col("rim_ya_conocida")))
        .otherwise(F.col("rim_proyectada")).alias("rim_valor"),
        F.lit("PREDICHA").alias("origen"),
        F.when(parcial, "PARCIAL").otherwise("MODELO").alias("detalle_origen"),
        "prob_cotiza", "rim_condicional",
        F.when(parcial, F.greatest(F.col("rim_esperada"), F.col("rim_ya_conocida")))
        .otherwise(F.col("rim_esperada")).alias("rim_esperada"),
        # en filas PARCIAL se conserva la prediccion pura del modelo para medir su error despues
        F.when(parcial, F.col("rim_proyectada")).otherwise(F.lit(None).cast("double")).alias("rim_predicha_previa"),
        F.greatest(F.col("n_pagadores_l1"), F.lit(1.0)).cast("int").alias("n_pagadores_esperados"),
        "tipo_afiliado",
        F.lit(periodo_snapshot).alias("periodo_snapshot"), F.lit(None).cast("int").alias("periodo_reemplazo"),
        F.lit(modelo.version).alias("version_modelo"), F.lit(fecha_calculo).alias("fecha_calculo")))
    reales = (features.filter(F.col("_completo") == 1).select(
        "afiliado_id", F.col("periodo_target").alias("periodo"), F.col("horizonte").cast("int"),
        F.col("rim_ya_conocida").alias("rim_valor"), F.lit("REAL").alias("origen"),
        F.lit("PAGO_ANTICIPADO").alias("detalle_origen"), F.lit(1.0).alias("prob_cotiza"),
        F.col("rim_ya_conocida").alias("rim_condicional"), F.col("rim_ya_conocida").alias("rim_esperada"),
        F.lit(None).cast("double").alias("rim_predicha_previa"),
        F.col("n_pagadores_ya_conocidos").cast("int").alias("n_pagadores_esperados"), "tipo_afiliado",
        F.lit(periodo_snapshot).alias("periodo_snapshot"), F.lit(None).cast("int").alias("periodo_reemplazo"),
        F.lit(modelo.version).alias("version_modelo"), F.lit(fecha_calculo).alias("fecha_calculo")))
    return predichas.unionByName(reales).select(*COLUMNAS_PROYECCION)
