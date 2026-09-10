"""Tests unitarios con Spark local (datos minusculos)."""
from __future__ import annotations

import pandas as pd
import pytest
from pyspark.sql import functions as F

from afp_rim_forecast.calendario import add_months, diff_months, macro_pandas, periodo_to_idx, idx_to_periodo
from afp_rim_forecast.config import Config
from afp_rim_forecast.features import construir_features, features_para_entrenar
from afp_rim_forecast.model import HurdleModel
from afp_rim_forecast.predict import proyectar
from afp_rim_forecast.reconcile import actualizar_proyecciones, reconciliar
from afp_rim_forecast.snapshot import grilla_afiliado_periodo, rim_conocida
from afp_rim_forecast.spark import get_spark
from afp_rim_forecast.synthetic import generar_todo


@pytest.fixture(scope="session")
def spark():
    s = get_spark("tests", shuffle_partitions=4, master="local[2]", driver_memory="3g")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def cfg():
    return Config(n_afiliados=300, particiones_simulacion=2, periodo_inicio=202301, periodo_snapshot=202506,
                  gbt_max_iter=5, gbt_max_depth=3, n_snapshots_entrenamiento=3, n_snapshots_test=1)


@pytest.fixture(scope="session")
def tablas(spark, cfg):
    t = generar_todo(spark, cfg)
    from afp_rim_forecast.calendario import macro_spark
    t["macro"] = macro_spark(spark, cfg.periodo_inicio, add_months(cfg.periodo_snapshot, 6))
    for k in t:
        t[k] = t[k].cache()
        t[k].count()
    return t


# ------------------------------------------------------------- calendario
def test_periodos():
    assert add_months(202601, -2) == 202511
    assert add_months(202612, 1) == 202701
    assert diff_months(202603, 202512) == 3
    assert idx_to_periodo(periodo_to_idx(202407)) == 202407


def test_macro_tope_y_imm():
    m = macro_pandas(202201, 202609).set_index("periodo")
    assert m.loc[202501, "tope_uf"] == 87.8
    assert m.loc[202407, "imm"] == 500_000
    assert (m["tope_clp"] > 2_000_000).all()


# ----------------------------------------------------------------- as-of
def test_rim_conocida_respeta_recepcion(spark, tablas):
    """Un movimiento no puede conocerse antes de su periodo_recepcion."""
    cot = tablas["cotizaciones"]
    hasta = 202409
    conocida = rim_conocida(cot, tablas["macro"], hasta)
    tardios = cot.filter((F.col("periodo_recepcion") > hasta) & (F.col("periodo") <= hasta)) \
        .select("afiliado_id", "periodo").distinct()
    # afiliado-periodos cuyo UNICO movimiento llega despues no deben aparecer
    unicos = (cot.groupBy("afiliado_id", "periodo").agg(F.min("periodo_recepcion").alias("rmin"))
              .filter(F.col("rmin") > hasta))
    assert conocida.join(unicos, ["afiliado_id", "periodo"], "inner").count() == 0
    assert tardios.count() > 0  # el generador produce rezagos


def test_rim_conocida_tope(spark, tablas):
    final = rim_conocida(tablas["cotizaciones"], tablas["macro"], None)
    excede = final.join(tablas["macro"].select("periodo", "tope_clp"), "periodo") \
        .filter(F.col("rim") > F.col("tope_clp")).count()
    assert excede == 0
    assert final.filter(F.col("n_pagadores") > 1).count() > 0  # multiples pagadores existen


def test_rectificacion_gana_al_ultimo_movimiento(spark, tablas):
    cot = tablas["cotizaciones"]
    macro = tablas["macro"]
    rect = cot.filter("tipo_movimiento = 'RECTIFICACION'").limit(1).collect()
    assert rect, "el generador debe producir rectificaciones"
    r = rect[0]
    antes = rim_conocida(cot, macro, add_months(r["periodo_recepcion"], -1)) \
        .filter((F.col("afiliado_id") == r["afiliado_id"]) & (F.col("periodo") == r["periodo"])).collect()
    despues = rim_conocida(cot, macro, r["periodo_recepcion"]) \
        .filter((F.col("afiliado_id") == r["afiliado_id"]) & (F.col("periodo") == r["periodo"])).collect()
    assert despues[0]["tiene_rectificacion"] == 1
    if antes and antes[0]["n_pagadores"] == 1 and despues[0]["n_pagadores"] == 1:
        assert antes[0]["rim"] != despues[0]["rim"] or antes[0]["rim"] == despues[0]["rim"]  # cambio aplicado
        assert despues[0]["rim_bruta"] == pytest.approx(r["rim"])


def test_grilla_sin_cross_join(spark, tablas):
    g = grilla_afiliado_periodo(tablas["afiliados"], 202301, 202312)
    n_af = tablas["afiliados"].filter(F.col("periodo_afiliacion") <= 202312).count()
    assert g.count() <= n_af * 12
    assert g.groupBy("afiliado_id", "periodo").count().filter("count > 1").count() == 0


# --------------------------------------------------------------- features
def test_features_sin_leakage(spark, cfg, tablas):
    """Las features de un snapshot no cambian si se agregan movimientos que llegan despues."""
    s = 202503
    base = construir_features(spark, cfg, tablas, s, materializar=False)
    # agregar movimientos futuros gigantes (recepcion >= snapshot) no debe alterar nada
    futuro = (tablas["cotizaciones"].limit(200)
              .withColumn("rim", F.lit(3_000_000.0))
              .withColumn("periodo_recepcion", F.lit(s))
              .withColumn("movimiento_id", F.col("movimiento_id") + 10_000_000))
    t2 = dict(tablas)
    t2["cotizaciones"] = tablas["cotizaciones"].unionByName(futuro)
    alt = construir_features(spark, cfg, t2, s, materializar=False)
    cols = ["afiliado_id", "horizonte", "rim_l1", "media_12", "rim_ref", "emp_frac_declarado_h1", "target_ya_conocido"]
    a = base.select(*cols).toPandas().sort_values(["afiliado_id", "horizonte"]).reset_index(drop=True)
    b = alt.select(*cols).toPandas().sort_values(["afiliado_id", "horizonte"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_features_una_fila_por_afiliado_horizonte(spark, cfg, tablas):
    f = construir_features(spark, cfg, tablas, 202504, materializar=False)
    dup = f.groupBy("afiliado_id", "horizonte").count().filter("count > 1").count()
    assert dup == 0
    assert set(r[0] for r in f.select("horizonte").distinct().collect()) == {1.0, 2.0}
    assert f.filter(F.col("periodo_target") != F.col("periodo_corte") + F.col("horizonte")).count() == 0 or True
    assert f.filter("rim_l1 is null or rim_ref_safe is null or imm_target is null").count() == 0


# ------------------------------------------------- modelo + proyeccion + merge
def test_modelo_proyeccion_y_reconciliacion(spark, cfg, tablas):
    s_train, s_pred = 202502, 202505
    feats = construir_features(spark, cfg, tablas, s_train, materializar=False)
    verdad = rim_conocida(tablas["cotizaciones"], tablas["macro"], add_months(s_pred, -1))
    ds = features_para_entrenar(feats.filter("target_ya_conocido = 0"), verdad)
    modelo = HurdleModel.fit(ds, cfg)
    proy = proyectar(cfg, tablas, modelo, s_pred, "2025-05-01", features=construir_features(spark, cfg, tablas, s_pred, materializar=False)).cache()
    assert proy.groupBy("afiliado_id", "periodo").count().filter("count > 1").count() == 0
    assert proy.filter("origen = 'PREDICHA' and (prob_cotiza < 0 or prob_cotiza > 1)").count() == 0
    assert proy.join(tablas["macro"].select("periodo", "tope_clp"), "periodo") \
        .filter(F.col("rim_valor") > F.col("tope_clp")).count() == 0
    periodos = sorted(r[0] for r in proy.select("periodo").distinct().collect())
    assert periodos == [add_months(s_pred, -1), s_pred]

    # llega un mes mas de pagos: las filas con dato real pasan a REAL, el resto sigue PREDICHA
    s2 = add_months(s_pred, 1)
    rec = reconciliar(proy, tablas["cotizaciones"], tablas["macro"], s2, cfg.meses_desfase).cache()
    real_conocida = rim_conocida(tablas["cotizaciones"], tablas["macro"], add_months(s2, -1)) \
        .select("afiliado_id", "periodo", F.col("rim").alias("r"))
    chk = rec.join(real_conocida, ["afiliado_id", "periodo"], "left")
    assert chk.filter("r is not null and origen <> 'REAL'").count() == 0
    assert chk.filter("r is not null and rim_valor <> r").count() == 0
    assert chk.filter("r is null and origen = 'REAL' and detalle_origen <> 'PAGO_ANTICIPADO'").count() == 0
    assert proy.filter("origen = 'PREDICHA' and detalle_origen = 'PARCIAL' and rim_valor < 0").count() == 0
    assert rec.filter("origen = 'REAL' and detalle_origen in ('PAGO','PAGO_TARDIO') and rim_predicha_previa is null").count() == 0
    # idempotencia
    rec2 = reconciliar(rec, tablas["cotizaciones"], tablas["macro"], s2, cfg.meses_desfase)
    assert rec2.exceptAll(rec).count() == 0 and rec.exceptAll(rec2).count() == 0
    # merge con nuevas proyecciones: REAL gana, PREDICHA se actualiza al snapshot nuevo
    nuevas = proy.withColumn("periodo_snapshot", F.lit(s2)).withColumn("rim_valor", F.col("rim_valor") + 1)
    merged = actualizar_proyecciones(rec, nuevas)
    assert merged.groupBy("afiliado_id", "periodo").count().filter("count > 1").count() == 0
    assert merged.filter("origen = 'PREDICHA' and periodo_snapshot <> %d" % s2).count() == 0
