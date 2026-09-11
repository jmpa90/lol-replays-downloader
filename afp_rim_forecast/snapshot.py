"""Vista "as-of": que RIM conoce la AFP en un snapshot dado.

Regla de negocio:
  * Un movimiento se conoce si `periodo_recepcion <= hasta_recepcion`.
  * Por cada (afiliado, periodo, entidad_pagadora, empleador) vale el ULTIMO movimiento
    que informa monto (DECLARACION o RECTIFICACION), ordenado por periodo_recepcion y
    movimiento_id. Una DNP (declaracion y no pago) informa RIM igual; el PAGO_DNP
    posterior no cambia el monto.
  * La RIM del afiliado en el mes es la suma sobre pagadores, topada al tope imponible
    del periodo.
Esta misma funcion define la "verdad final" cuando hasta_recepcion es None.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F

from .calendario import col_idx_to_periodo, col_periodo_to_idx, periodo_to_idx


def rim_conocida(cotizaciones: DataFrame, macro: DataFrame, hasta_recepcion: int | None) -> DataFrame:
    """RIM total conocida por (afiliado, periodo) con la informacion recibida hasta `hasta_recepcion`."""
    mov = cotizaciones.filter(F.col("tipo_movimiento").isin("DECLARACION", "RECTIFICACION"))
    if hasta_recepcion is not None:
        mov = mov.filter(F.col("periodo_recepcion") <= hasta_recepcion)
    # PAGO_DNP conocidos hasta el snapshot regularizan la DNP
    pagos = cotizaciones.filter(F.col("tipo_movimiento") == "PAGO_DNP")
    if hasta_recepcion is not None:
        pagos = pagos.filter(F.col("periodo_recepcion") <= hasta_recepcion)
    pagos = pagos.select("afiliado_id", "periodo", "empleador_id").distinct().withColumn("regularizada", F.lit(1))

    clave = ["afiliado_id", "periodo", "entidad_pagadora", "empleador_id"]
    w = W.partitionBy(*clave).orderBy(F.col("periodo_recepcion").desc(), F.col("movimiento_id").desc())
    # el flag DNP sale de la fila vigente (ya filtrada as-of), nunca de movimientos aun no recibidos
    vigente = (mov.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")
               .join(pagos, ["afiliado_id", "periodo", "empleador_id"], "left")
               .withColumn("tiene_dnp", F.when((F.col("estado_pago") == "DNP") & F.col("regularizada").isNull(), 1)
                           .otherwise(0)))

    w_emp = W.partitionBy("afiliado_id", "periodo").orderBy(F.col("rim").desc(), F.col("empleador_id"))
    agg = (vigente
           .withColumn("_rk", F.row_number().over(w_emp))
           .groupBy("afiliado_id", "periodo")
           .agg(F.sum("rim").alias("rim_bruta"),
                # count(distinct a, b) descarta filas con nulos (SII/AFILIADO no tienen empleador)
                F.countDistinct(F.concat_ws("#", F.col("entidad_pagadora"),
                                            F.coalesce(F.col("empleador_id").cast("string"), F.lit("NA")))).alias("n_pagadores"),
                F.max(F.when(F.col("_rk") == 1, F.col("empleador_id"))).alias("empleador_principal"),
                F.max(F.when(F.col("_rk") == 1, F.col("entidad_pagadora"))).alias("pagador_principal"),
                F.max((F.col("entidad_pagadora") == "SUBSIDIO").cast("int")).alias("tiene_subsidio"),
                F.max("tiene_dnp").alias("tiene_dnp"),
                F.max((F.col("tipo_movimiento") == "RECTIFICACION").cast("int")).alias("tiene_rectificacion"),
                F.max("periodo_recepcion").alias("periodo_recepcion_max"),
                F.min("periodo_recepcion").alias("periodo_recepcion_min")))
    tope = F.broadcast(macro.select("periodo", "tope_clp"))
    return (agg.join(tope, "periodo", "left")
            .withColumn("rim", F.round(F.least(F.col("rim_bruta"), F.col("tope_clp")), 0))
            .drop("tope_clp"))


def grilla_afiliado_periodo(afiliados: DataFrame, periodo_ini: int, periodo_fin: int) -> DataFrame:
    """Grilla densa afiliado x mes (desde su afiliacion) sin cross join: sequence + explode por fila."""
    idx_ini, idx_fin = periodo_to_idx(periodo_ini), periodo_to_idx(periodo_fin)
    return (afiliados.select("afiliado_id", "periodo_afiliacion")
            .withColumn("_ini", F.greatest(col_periodo_to_idx(F.col("periodo_afiliacion")), F.lit(idx_ini)))
            .filter(F.col("_ini") <= idx_fin)
            .withColumn("periodo_idx", F.explode(F.sequence(F.col("_ini"), F.lit(idx_fin))))
            .withColumn("periodo", col_idx_to_periodo(F.col("periodo_idx")).cast("int"))
            .select("afiliado_id", "periodo", "periodo_idx"))


def serie_conocida(afiliados: DataFrame, cotizaciones: DataFrame, macro: DataFrame,
                   periodo_ini: int, periodo_corte: int, hasta_recepcion: int) -> DataFrame:
    """Serie mensual densa (meses sin cotizacion = 0) conocida en el snapshot, hasta periodo_corte."""
    conocida = rim_conocida(cotizaciones, macro, hasta_recepcion)
    grilla = grilla_afiliado_periodo(afiliados, periodo_ini, periodo_corte)
    return (grilla.join(conocida, ["afiliado_id", "periodo"], "left")
            .fillna({"rim": 0.0, "rim_bruta": 0.0, "n_pagadores": 0, "tiene_subsidio": 0, "tiene_dnp": 0,
                     "tiene_rectificacion": 0}))


def conocidas_meses_abiertos(cotizaciones: DataFrame, macro: DataFrame, periodo_corte: int,
                             hasta_recepcion: int) -> DataFrame:
    """Filas ya conocidas para meses posteriores al corte (empleadores que pagan anticipado)."""
    return (rim_conocida(cotizaciones, macro, hasta_recepcion)
            .filter(F.col("periodo") > periodo_corte)
            .select("afiliado_id", "periodo", "rim", "n_pagadores", "periodo_recepcion_max"))
