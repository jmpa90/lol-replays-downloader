"""Feature engineering "as-of" para un snapshot.

Todas las features de una fila se calculan SOLO con informacion recibida hasta
`hasta_recepcion = snapshot - 1` (fin del mes anterior al que corre el modelo).
Esto se garantiza construyendo primero la serie conocida (snapshot.py) y aplicando
window functions sobre ella; nunca se mira la tabla de verdad.

Salida: una fila por (afiliado, horizonte) con `periodo_target = periodo_corte + h`.
En produccion esta tabla se calcula una vez al mes y se persiste particionada por
`periodo_snapshot`; el set de entrenamiento es la union de snapshots pasados con
el target real que llego despues (nunca se recalculan features "hacia atras").
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window as W
from pyspark.sql import functions as F

from .calendario import add_months, col_periodo_to_idx, periodo_to_idx
from .config import Config
from .snapshot import conocidas_meses_abiertos, serie_conocida

N_LAGS = 12

FEATURES_NUMERICAS = (
    [f"rim_l{k}" for k in range(1, N_LAGS + 1)]
    + ["media_3", "media_6", "media_12", "std_6", "max_12", "min_12",
       "n_cotiza_3", "n_cotiza_6", "n_cotiza_12", "meses_historia",
       "meses_desde_ult_cotiza", "meses_desde_cambio", "ratio_l1_media12", "tendencia_3_12",
       "rim_ref", "rim_ref_sobre_imm", "rim_ref_sobre_tope", "rim_mismo_mes_ly", "ratio_ly_ref",
       "meses_con_empleador", "n_empleadores_12", "n_pagadores_l1", "tiene_subsidio_l1", "tiene_dnp_l1",
       "dias_licencia_l1", "dias_licencia_l2", "licencia_en_curso",
       "afc_termino_l1", "afc_termino_l2", "afc_inicio_l1", "afc_inicio_l2",
       "meses_desde_afc_termino", "meses_desde_afc_inicio",
       "emp_n_trabajadores", "emp_crecimiento_12", "emp_rim_mediana", "emp_tasa_oportuno",
       "emp_frac_declarado_corte", "emp_frac_declarado_h1", "sin_declaracion_con_emp_activo",
       "edad", "meses_desde_afiliacion", "region",
       "horizonte", "mes_target", "es_enero", "es_marzo", "es_abril", "es_julio", "es_sept", "es_dic",
       "imm_target", "tope_target", "ratio_imm_target_corte", "ipc_12m_corte", "desempleo_corte"]
)
FEATURES_CATEGORICAS = ["tipo_afiliado", "sexo", "nivel_educacional", "rubro", "tamano_empleador",
                        "tipo_contrato"]


def _features_serie(serie: DataFrame, periodo_corte: int) -> DataFrame:
    """Features de historia propia, evaluadas en el mes de corte (una fila por afiliado)."""
    w = W.partitionBy("afiliado_id").orderBy("periodo_idx")
    w_all = w.rowsBetween(W.unboundedPreceding, 0)
    df = serie.withColumn("cotiza", (F.col("rim") > 0).cast("int"))
    for k in range(1, N_LAGS + 1):
        df = df.withColumn(f"rim_l{k}", F.lag("rim", k - 1).over(w))
    df = (df
          .withColumn("media_3", F.avg("rim").over(w.rowsBetween(-2, 0)))
          .withColumn("media_6", F.avg("rim").over(w.rowsBetween(-5, 0)))
          .withColumn("media_12", F.avg("rim").over(w.rowsBetween(-11, 0)))
          .withColumn("std_6", F.coalesce(F.stddev("rim").over(w.rowsBetween(-5, 0)), F.lit(0.0)))
          .withColumn("max_12", F.max("rim").over(w.rowsBetween(-11, 0)))
          .withColumn("min_12", F.min("rim").over(w.rowsBetween(-11, 0)))
          .withColumn("n_cotiza_3", F.sum("cotiza").over(w.rowsBetween(-2, 0)))
          .withColumn("n_cotiza_6", F.sum("cotiza").over(w.rowsBetween(-5, 0)))
          .withColumn("n_cotiza_12", F.sum("cotiza").over(w.rowsBetween(-11, 0)))
          .withColumn("meses_historia", F.count("rim").over(w_all))
          .withColumn("_ult_cotiza_idx", F.max(F.when(F.col("cotiza") == 1, F.col("periodo_idx"))).over(w_all))
          .withColumn("rim_ref", F.last(F.when(F.col("rim") > 0, F.col("rim")), ignorenulls=True).over(w_all))
          .withColumn("empleador_ref", F.last(F.when(F.col("rim") > 0, F.col("empleador_principal")),
                                              ignorenulls=True).over(w_all))
          .withColumn("_rim_prev", F.lag("rim", 1).over(w))
          .withColumn("_cambio", F.when(F.col("_rim_prev").isNull(), 1)
                      .when((F.col("_rim_prev") == 0) != (F.col("rim") == 0), 1)
                      .when(F.abs(F.col("rim") - F.col("_rim_prev")) > 0.02 * F.col("_rim_prev"), 1).otherwise(0))
          .withColumn("_ult_cambio_idx", F.max(F.when(F.col("_cambio") == 1, F.col("periodo_idx"))).over(w_all))
          # racha de meses consecutivos con el mismo empleador principal (terminando en el corte)
          .withColumn("_emp_prev", F.lag("empleador_principal", 1).over(w))
          .withColumn("_corte_emp", F.when(F.col("empleador_principal").isNull()
                                           | (F.col("empleador_principal") != F.col("_emp_prev"))
                                           | F.col("_emp_prev").isNull(), F.col("periodo_idx")))
          .withColumn("_ini_racha_emp", F.max("_corte_emp").over(w_all))
          .withColumn("n_empleadores_12", F.size(F.array_distinct(F.collect_list("empleador_principal")
                                                                  .over(w.rowsBetween(-11, 0)))))
          )
    corte_idx = periodo_to_idx(periodo_corte)
    at_corte = (df.filter(F.col("periodo_idx") == corte_idx)
                .withColumn("meses_desde_ult_cotiza",
                            F.coalesce(F.lit(corte_idx) - F.col("_ult_cotiza_idx"), F.lit(99)))
                .withColumn("meses_desde_cambio", F.lit(corte_idx) - F.col("_ult_cambio_idx"))
                .withColumn("meses_con_empleador",
                            F.when(F.col("empleador_principal").isNull(), 0)
                            .otherwise(F.col("periodo_idx") - F.col("_ini_racha_emp") + 1))
                .withColumn("ratio_l1_media12", F.col("rim_l1") / (F.col("media_12") + 1.0))
                .withColumn("tendencia_3_12", F.col("media_3") / (F.col("media_12") + 1.0))
                .withColumn("n_pagadores_l1", F.col("n_pagadores"))
                .withColumn("tiene_subsidio_l1", F.col("tiene_subsidio"))
                .withColumn("tiene_dnp_l1", F.col("tiene_dnp"))
                .withColumn("empleador_actual", F.col("empleador_principal")))
    cols = (["afiliado_id", "rim_ref", "empleador_ref", "empleador_actual", "meses_desde_ult_cotiza",
             "meses_desde_cambio", "meses_con_empleador", "n_empleadores_12", "ratio_l1_media12",
             "tendencia_3_12", "n_pagadores_l1", "tiene_subsidio_l1", "tiene_dnp_l1"]
            + [f"rim_l{k}" for k in range(1, N_LAGS + 1)]
            + ["media_3", "media_6", "media_12", "std_6", "max_12", "min_12", "n_cotiza_3", "n_cotiza_6",
               "n_cotiza_12", "meses_historia"])
    return at_corte.select(*cols)


def _features_empleador(serie: DataFrame, cotizaciones: DataFrame, conocidas_abiertas: DataFrame,
                        periodo_corte: int, hasta_recepcion: int) -> DataFrame:
    """Features del empleador de referencia, calculadas con la serie conocida."""
    corte_idx = periodo_to_idx(periodo_corte)
    activos = serie.filter(F.col("empleador_principal").isNotNull())
    en_corte = (activos.filter(F.col("periodo_idx") == corte_idx)
                .groupBy("empleador_principal")
                .agg(F.count("*").alias("emp_n_trabajadores"),
                     F.expr("percentile_approx(rim, 0.5)").alias("emp_rim_mediana")))
    hace_12 = (activos.filter(F.col("periodo_idx") == corte_idx - 12)
               .groupBy("empleador_principal").agg(F.count("*").alias("_n_12")))
    # trabajadores del empleador en corte-1: cuantos ya tienen declaracion del mes de corte
    prev = activos.filter(F.col("periodo_idx") == corte_idx - 1).select("afiliado_id", "empleador_principal")
    decl_corte = (serie.filter((F.col("periodo_idx") == corte_idx) & (F.col("rim") > 0))
                  .select("afiliado_id", F.lit(1).alias("_decl")))
    frac_corte = (prev.join(decl_corte, "afiliado_id", "left")
                  .groupBy("empleador_principal")
                  .agg(F.avg(F.coalesce(F.col("_decl"), F.lit(0))).alias("emp_frac_declarado_corte")))
    # trabajadores en corte: cuantos ya tienen declaracion (anticipada) del mes corte+1
    act_corte = activos.filter(F.col("periodo_idx") == corte_idx).select("afiliado_id", "empleador_principal")
    decl_h1 = (conocidas_abiertas.filter(F.col("periodo") == add_months(periodo_corte, 1))
               .select("afiliado_id", F.lit(1).alias("_decl")))
    frac_h1 = (act_corte.join(decl_h1, "afiliado_id", "left")
               .groupBy("empleador_principal")
               .agg(F.avg(F.coalesce(F.col("_decl"), F.lit(0))).alias("emp_frac_declarado_h1")))
    # puntualidad historica del empleador (fraccion de declaraciones recibidas en m+1 o antes)
    ult12 = add_months(periodo_corte, -11)
    punt = (cotizaciones
            .filter((F.col("entidad_pagadora") == "EMPLEADOR") & (F.col("tipo_movimiento") == "DECLARACION")
                    & (F.col("periodo_recepcion") <= hasta_recepcion) & (F.col("periodo") >= ult12)
                    & (F.col("periodo") <= periodo_corte))
            .withColumn("_lag", col_periodo_to_idx(F.col("periodo_recepcion")) - col_periodo_to_idx(F.col("periodo")))
            .groupBy(F.col("empleador_id").alias("empleador_principal"))
            .agg(F.avg((F.col("_lag") <= 1).cast("double")).alias("emp_tasa_oportuno")))
    emp = (en_corte.join(hace_12, "empleador_principal", "left")
           .join(frac_corte, "empleador_principal", "left")
           .join(frac_h1, "empleador_principal", "left")
           .join(punt, "empleador_principal", "left")
           .withColumn("emp_crecimiento_12", F.col("emp_n_trabajadores") / (F.coalesce(F.col("_n_12"), F.lit(0)) + 1.0))
           .drop("_n_12")
           .withColumnRenamed("empleador_principal", "empleador_ref"))
    return emp


def _features_eventos(afc: DataFrame, licencias: DataFrame, periodo_corte: int, hasta_recepcion: int) -> DataFrame:
    corte_idx = periodo_to_idx(periodo_corte)
    afc_k = (afc.filter(F.col("periodo_recepcion") <= hasta_recepcion)
             .withColumn("_d", F.lit(corte_idx) - col_periodo_to_idx(F.col("periodo")))
             .filter(F.col("_d") >= 0)
             .groupBy("afiliado_id")
             .agg(F.max(F.when((F.col("tipo_evento") == "TERMINO") & (F.col("_d") == 0), 1).otherwise(0)).alias("afc_termino_l1"),
                  F.max(F.when((F.col("tipo_evento") == "TERMINO") & (F.col("_d") == 1), 1).otherwise(0)).alias("afc_termino_l2"),
                  F.max(F.when((F.col("tipo_evento") == "INICIO") & (F.col("_d") == 0), 1).otherwise(0)).alias("afc_inicio_l1"),
                  F.max(F.when((F.col("tipo_evento") == "INICIO") & (F.col("_d") == 1), 1).otherwise(0)).alias("afc_inicio_l2"),
                  F.min(F.when(F.col("tipo_evento") == "TERMINO", F.col("_d"))).alias("meses_desde_afc_termino"),
                  F.min(F.when(F.col("tipo_evento") == "INICIO", F.col("_d"))).alias("meses_desde_afc_inicio")))
    lic_k = (licencias.filter(F.col("periodo_recepcion") <= hasta_recepcion)
             .withColumn("_d", F.lit(corte_idx) - col_periodo_to_idx(F.col("periodo")))
             .filter(F.col("_d").isin(0, 1))
             .groupBy("afiliado_id")
             .agg(F.max(F.when(F.col("_d") == 0, F.col("dias_licencia")).otherwise(0)).alias("dias_licencia_l1"),
                  F.max(F.when(F.col("_d") == 1, F.col("dias_licencia")).otherwise(0)).alias("dias_licencia_l2"))
             .withColumn("licencia_en_curso", (F.col("dias_licencia_l1") >= 30).cast("int")))
    return afc_k, lic_k


def construir_features(spark: SparkSession, cfg: Config, tablas: dict[str, DataFrame],
                       periodo_snapshot: int, materializar: bool = True) -> DataFrame:
    """Dataset de features para un snapshot: una fila por (afiliado, horizonte).

    Con `materializar=True` el resultado queda cacheado (equivale a persistirlo en la
    tabla de features particionada por periodo_snapshot) y se liberan los intermedios.
    """
    corte = add_months(periodo_snapshot, -cfg.meses_desfase)
    hasta = add_months(periodo_snapshot, -1)
    afiliados, empleadores, macro = tablas["afiliados"], tablas["empleadores"], tablas["macro"]
    cotizaciones = tablas["cotizaciones"]

    # la serie conocida se reutiliza en varias features: se cachea para no recalcular rim_conocida
    serie = serie_conocida(afiliados, cotizaciones, macro, cfg.periodo_inicio, corte, hasta).cache()
    abiertas = conocidas_meses_abiertos(cotizaciones, macro, corte, hasta).cache()
    f_serie = _features_serie(serie, corte)
    f_emp = _features_empleador(serie, cotizaciones, abiertas, corte, hasta)
    f_afc, f_lic = _features_eventos(tablas["afc_eventos"], tablas["licencias"], corte, hasta)

    macro_p = macro.toPandas().set_index("periodo")
    m_corte = macro_p.loc[corte]
    dim_emp = F.broadcast(empleadores.select(F.col("empleador_id").alias("empleador_ref"), "rubro",
                                             "tamano_empleador", "region_empleador"))
    dim_af = afiliados.select("afiliado_id", "sexo", "edad_inicio", "region", "nivel_educacional",
                              "tipo_afiliado", "periodo_afiliacion")

    base = (dim_af.filter(col_periodo_to_idx(F.col("periodo_afiliacion")) <= periodo_to_idx(corte) + max(cfg.horizontes))
            .join(f_serie, "afiliado_id", "left")
            .join(f_emp, "empleador_ref", "left")
            .join(dim_emp, "empleador_ref", "left")
            .join(f_afc, "afiliado_id", "left")
            .join(f_lic, "afiliado_id", "left"))

    # tipo de contrato conocido: se infiere del ultimo evento AFC / no esta en cotizaciones.
    # En la simulacion no se expone al modelo el contrato latente; se usa proxy por antiguedad.
    base = base.withColumn("tipo_contrato", F.when(F.col("empleador_actual").isNull(), "SIN_EMPLEADOR")
                           .when(F.col("meses_con_empleador") < 12, "RECIENTE").otherwise("ANTIGUO"))

    filas = []
    for h in cfg.horizontes:
        target = add_months(corte, h)
        m_t = macro_p.loc[target]
        mes_t = target % 100
        ly = F.col(f"rim_l{13 - h}") if 13 - h <= N_LAGS else F.lit(None)   # valor en target-12
        filas.append(
            base.withColumn("horizonte", F.lit(h))
            .withColumn("periodo_target", F.lit(target))
            .withColumn("mes_target", F.lit(mes_t))
            .withColumn("rim_mismo_mes_ly", ly)
            .withColumn("imm_target", F.lit(float(m_t["imm"])))
            .withColumn("tope_target", F.lit(float(m_t["tope_clp"])))
            .withColumn("ratio_imm_target_corte", F.lit(float(m_t["imm"]) / float(m_corte["imm"])))
            .withColumn("ipc_12m_corte", F.lit(float(m_corte["ipc_12m"])))
            .withColumn("desempleo_corte", F.lit(float(m_corte["desempleo"])))
            .withColumn("imm_var_12m_target", F.lit(float(m_t["imm_var_12m"])))
        )
    df = filas[0]
    for extra in filas[1:]:
        df = df.unionByName(extra)

    ya_conocido = abiertas.select("afiliado_id", F.col("periodo").alias("periodo_target"),
                                  F.col("rim").alias("rim_ya_conocida"))
    df = (df.join(ya_conocido, ["afiliado_id", "periodo_target"], "left")
          .withColumn("target_ya_conocido", F.col("rim_ya_conocida").isNotNull().cast("int"))
          .withColumn("periodo_snapshot", F.lit(periodo_snapshot))
          .withColumn("periodo_corte", F.lit(corte))
          .withColumn("edad", F.col("edad_inicio") + F.floor((F.lit(periodo_to_idx(corte)) + F.col("horizonte")
                                                             - F.lit(periodo_to_idx(cfg.periodo_inicio))) / 12))
          .withColumn("meses_desde_afiliacion", F.lit(periodo_to_idx(corte)) - col_periodo_to_idx(F.col("periodo_afiliacion")))
          .withColumn("rim_ref_safe", F.coalesce(F.col("rim_ref"), F.col("imm_target")))
          .withColumn("rim_ref_sobre_imm", F.col("rim_ref_safe") / F.col("imm_target"))
          .withColumn("rim_ref_sobre_tope", F.col("rim_ref_safe") / F.col("tope_target"))
          .withColumn("ratio_ly_ref", F.col("rim_mismo_mes_ly") / F.col("rim_ref_safe"))
          .withColumn("sin_declaracion_con_emp_activo",
                      F.when((F.col("rim_l1") == 0) & (F.col("rim_l2") > 0)
                             & (F.col("emp_frac_declarado_corte") >= 0.8), 1).otherwise(0))
          .withColumn("es_enero", (F.col("mes_target") == 1).cast("int"))
          .withColumn("es_marzo", (F.col("mes_target") == 3).cast("int"))
          .withColumn("es_abril", (F.col("mes_target") == 4).cast("int"))
          .withColumn("es_julio", (F.col("mes_target") == 7).cast("int"))
          .withColumn("es_sept", (F.col("mes_target") == 9).cast("int"))
          .withColumn("es_dic", (F.col("mes_target") == 12).cast("int"))
          .withColumn("region", F.col("region").cast("double")))

    # rellenos: sin historia => 0 en lags/rolling; eventos ausentes => 0; distancias => 99
    rellenos = {c: 0.0 for c in [f"rim_l{k}" for k in range(1, N_LAGS + 1)]
                + ["media_3", "media_6", "media_12", "std_6", "max_12", "min_12", "rim_mismo_mes_ly", "ratio_ly_ref",
                   "ratio_l1_media12", "tendencia_3_12", "emp_frac_declarado_corte", "emp_frac_declarado_h1",
                   "emp_crecimiento_12", "emp_rim_mediana", "emp_tasa_oportuno"]}
    rellenos.update({c: 0 for c in ["n_cotiza_3", "n_cotiza_6", "n_cotiza_12", "meses_historia", "meses_con_empleador",
                                    "n_empleadores_12", "n_pagadores_l1", "tiene_subsidio_l1", "tiene_dnp_l1",
                                    "dias_licencia_l1", "dias_licencia_l2", "licencia_en_curso",
                                    "afc_termino_l1", "afc_termino_l2", "afc_inicio_l1", "afc_inicio_l2",
                                    "emp_n_trabajadores"]})
    rellenos.update({"meses_desde_ult_cotiza": 99, "meses_desde_cambio": 99, "meses_desde_afc_termino": 99,
                     "meses_desde_afc_inicio": 99, "rim_ref": 0.0,
                     "rubro": "SIN_EMPLEADOR", "tamano_empleador": "SIN_EMPLEADOR"})
    df = df.fillna(rellenos)
    for c in FEATURES_NUMERICAS:
        df = df.withColumn(c, F.col(c).cast("double"))
    if materializar:
        df = df.cache()
        df.count()
        serie.unpersist()
        abiertas.unpersist()
    return df


def features_para_entrenar(features: DataFrame, verdad: DataFrame) -> DataFrame:
    """Une las features de un snapshot con la verdad final del periodo target."""
    y = verdad.select("afiliado_id", F.col("periodo").alias("periodo_target"), F.col("rim").alias("rim_real"))
    return (features.join(y, ["afiliado_id", "periodo_target"], "left")
            .fillna({"rim_real": 0.0})
            .withColumn("y_cls", (F.col("rim_real") > 0).cast("double"))
            .withColumn("y_reg", F.log(F.col("rim_real") / F.col("rim_ref_safe"))))
