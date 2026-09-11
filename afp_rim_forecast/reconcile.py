"""Reconciliacion: reemplazar la RIM PREDICHA por la REAL cuando llega el pago.

Ciclo mensual (snapshot nuevo = T+1):
  1. `reconciliar`:
     * PREDICHA cuyo periodo ya tiene TODOS los pagadores esperados -> REAL (detalle PAGO si llego
       en m+1, PAGO_TARDIO si tardo mas, DECLARADA_DNP si el empleador declaro y no pago).
       (En jerga AFP "rezago" es otra cosa: una cotizacion recibida que no pudo imputarse.)
     * PREDICHA con solo parte de los pagadores (multiempleo, subsidio de licencia pendiente) ->
       sigue PREDICHA con detalle PARCIAL y rim_valor = max(prediccion, lo ya recibido); pasa a REAL
       cuando se completa o cuando el periodo cierra.
     * REAL cuyo monto cambio -> se actualiza (RECTIFICACION si hubo rectificacion de planilla,
       COMPLEMENTO si llego otro pagador).
     * Periodo cerrado (mas alla de la ventana de pagos tardios) sin cotizacion -> 0 con detalle
       SIN_COTIZACION; si despues llega un pago tardio vuelve a actualizarse (PAGO_TARDIO).
       Para independientes el cierre espera a la Operacion Renta (julio del anio siguiente).
     Siempre se conserva `rim_predicha_previa` (la prediccion pura) para medir el error del modelo.
  2. `actualizar_proyecciones`: las filas que siguen PREDICHA se reemplazan por la nueva
     proyeccion (el h=2 del mes pasado pasa a ser h=1 con mas informacion).

Equivalente en Delta Lake (produccion):

    MERGE INTO rim_proyectada t
    USING rim_real_nueva s
      ON t.afiliado_id = s.afiliado_id AND t.periodo = s.periodo
    WHEN MATCHED AND t.origen = 'PREDICHA' AND s.n_pagadores >= t.n_pagadores_esperados THEN UPDATE SET
         rim_predicha_previa = coalesce(t.rim_predicha_previa, t.rim_valor), rim_valor = s.rim,
         origen = 'REAL', detalle_origen = s.detalle, periodo_reemplazo = :snapshot
    WHEN MATCHED AND t.origen = 'PREDICHA' THEN UPDATE SET
         rim_valor = greatest(t.rim_valor, s.rim), detalle_origen = 'PARCIAL'
    WHEN MATCHED AND t.origen = 'REAL' AND t.rim_valor <> s.rim THEN UPDATE SET
         rim_valor = s.rim, detalle_origen = 'RECTIFICACION', periodo_reemplazo = :snapshot
    WHEN NOT MATCHED THEN INSERT (...) VALUES (...)

La version DataFrame de abajo es idempotente: reprocesar el mismo snapshot no cambia nada.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F

from .calendario import add_months, col_periodo_to_idx
from .predict import COLUMNAS_PROYECCION
from .snapshot import rim_conocida


def reconciliar(proyecciones: DataFrame, cotizaciones: DataFrame, macro: DataFrame,
                periodo_snapshot_nuevo: int, meses_desfase: int = 2, meses_cierre: int = 4) -> DataFrame:
    """Devuelve la tabla de proyecciones con los reemplazos aplicados."""
    hasta = add_months(periodo_snapshot_nuevo, -1)
    real = (rim_conocida(cotizaciones, macro, hasta)
            .select("afiliado_id", "periodo", F.col("rim").alias("_rim_real"),
                    F.col("n_pagadores").alias("_n_pag"),
                    (col_periodo_to_idx(F.col("periodo_recepcion_min")) - col_periodo_to_idx(F.col("periodo"))).alias("_lag_min"),
                    F.col("tiene_rectificacion").alias("_rect"), F.col("tiene_dnp").alias("_dnp")))
    df = proyecciones.join(real, ["afiliado_id", "periodo"], "left")

    # un periodo esta "cerrado" cuando ya paso la ventana normal de pagos tardios;
    # los independientes cotizan via Operacion Renta: cierran en julio del anio siguiente
    edad_periodo = col_periodo_to_idx(F.lit(periodo_snapshot_nuevo)) - col_periodo_to_idx(F.col("periodo"))
    cierre_indep = F.lit(periodo_snapshot_nuevo) >= (F.floor(F.col("periodo") / 100) + 1) * 100 + 7
    periodo_cerrado = F.when(F.col("tipo_afiliado") == "INDEPENDIENTE", cierre_indep) \
        .otherwise(edad_periodo >= (meses_desfase + meses_cierre))

    llego = F.col("_rim_real").isNotNull()
    completo = F.col("_n_pag") >= F.greatest(F.coalesce(F.col("n_pagadores_esperados"), F.lit(1)), F.lit(1))
    es_pred = F.col("origen") == "PREDICHA"
    pasa_a_real = llego & es_pred & (completo | periodo_cerrado)
    parcial = llego & es_pred & ~completo & ~periodo_cerrado
    rectifica = llego & (F.col("origen") == "REAL") & (F.col("rim_valor") != F.col("_rim_real"))
    cierra_en_cero = ~llego & es_pred & periodo_cerrado
    cambia = pasa_a_real | rectifica | cierra_en_cero

    detalle_nuevo = (F.when(rectifica & (F.col("detalle_origen") == "SIN_COTIZACION"), "PAGO_TARDIO")
                     .when(rectifica & (F.col("_rect") == 1), "RECTIFICACION")
                     .when(rectifica, "COMPLEMENTO")
                     .when(cierra_en_cero, "SIN_COTIZACION")
                     .when(pasa_a_real & (F.col("_dnp") == 1), "DECLARADA_DNP")
                     .when(pasa_a_real & (F.col("_lag_min") <= 1), "PAGO")
                     .when(pasa_a_real, "PAGO_TARDIO"))
    out = (df
           .withColumn("rim_predicha_previa",
                       F.when((pasa_a_real | cierra_en_cero | parcial) & F.col("rim_predicha_previa").isNull(),
                              F.col("rim_valor")).otherwise(F.col("rim_predicha_previa")))
           .withColumn("periodo_reemplazo", F.when(cambia, F.lit(periodo_snapshot_nuevo)).otherwise(F.col("periodo_reemplazo")))
           .withColumn("detalle_origen", F.when(cambia, detalle_nuevo).when(parcial, "PARCIAL")
                       .otherwise(F.col("detalle_origen")))
           .withColumn("rim_valor", F.when(pasa_a_real | rectifica, F.col("_rim_real"))
                       .when(parcial, F.greatest(F.col("rim_valor"), F.col("_rim_real")))
                       .when(cierra_en_cero, F.lit(0.0)).otherwise(F.col("rim_valor")))
           .withColumn("origen", F.when(cambia, "REAL").otherwise(F.col("origen")))
           .drop("_rim_real", "_n_pag", "_lag_min", "_rect", "_dnp"))
    return out.select(*COLUMNAS_PROYECCION)


def actualizar_proyecciones(existentes: DataFrame, nuevas: DataFrame) -> DataFrame:
    """MERGE de proyecciones: una fila por (afiliado, periodo); REAL gana a PREDICHA y,
    entre PREDICHAS, gana el snapshot mas reciente. Idempotente."""
    todas = existentes.select(*COLUMNAS_PROYECCION).unionByName(nuevas.select(*COLUMNAS_PROYECCION))
    w = W.partitionBy("afiliado_id", "periodo").orderBy(
        F.when(F.col("origen") == "REAL", 0).otherwise(1), F.col("periodo_snapshot").desc())
    return todas.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def reporte_reemplazos(proyecciones: DataFrame, periodo_snapshot_nuevo: int) -> pd.DataFrame:
    """Error del modelo en las filas reemplazadas en este snapshot (monitoreo continuo)."""
    r = proyecciones.filter((F.col("periodo_reemplazo") == periodo_snapshot_nuevo)
                            & F.col("rim_predicha_previa").isNotNull())
    e = F.col("rim_predicha_previa") - F.col("rim_valor")
    return (r.groupBy("periodo", "tipo_afiliado", "detalle_origen")
            .agg(F.count("*").alias("n"), F.avg(F.abs(e)).alias("mae"),
                 (F.sum(F.abs(e)) / F.sum(F.abs(F.col("rim_valor")))).alias("wape"),
                 ((F.sum("rim_predicha_previa") - F.sum("rim_valor")) / F.sum("rim_valor")).alias("sesgo_agregado"),
                 F.avg(((F.col("rim_predicha_previa") > 0) == (F.col("rim_valor") > 0)).cast("double")).alias("exactitud_cero"))
            .orderBy("periodo", "tipo_afiliado", "detalle_origen").toPandas())
