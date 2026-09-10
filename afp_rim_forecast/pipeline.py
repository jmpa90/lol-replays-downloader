"""Orquestacion end-to-end (CLI).

    python -m afp_rim_forecast.pipeline demo --n-afiliados 5000 --salida /tmp/afp_rim_forecast

Pasos del `demo`:
  1. generar datos sinteticos y persistirlos en parquet (fuentes/)
  2. construir features para una ventana de snapshots (features/, particionado por periodo_snapshot)
  3. backtest: entrenar con snapshots antiguos, evaluar en snapshots recientes vs baselines
  4. entrenar el modelo final con todos los snapshots elegibles y proyectar los 2 meses abiertos
  5. simular el ciclo mensual siguiente: llegan pagos -> reconciliar (PREDICHA -> REAL) y re-proyectar
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .calendario import add_months, macro_spark, periodo_to_idx
from .config import Config
from .evaluate import agregar_baselines, metricas, metricas_clasificador, tramo_renta
from .features import construir_features, features_para_entrenar
from .model import HurdleModel
from .predict import proyectar
from .reconcile import actualizar_proyecciones, reconciliar, reporte_reemplazos
from .snapshot import rim_conocida
from .spark import get_spark
from .synthetic import generar_todo

FUENTES = ["afiliados", "empleadores", "verdad_mensual", "cotizaciones", "licencias", "afc_eventos", "macro"]


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ datos
def generar_fuentes(spark: SparkSession, cfg: Config, t0: float) -> dict[str, DataFrame]:
    tablas = generar_todo(spark, cfg)
    # la macro debe cubrir los meses target posteriores al snapshot
    tablas["macro"] = macro_spark(spark, cfg.periodo_inicio, add_months(cfg.periodo_snapshot, 6))
    for nombre in FUENTES:
        ruta = f"{cfg.ruta_salida}/fuentes/{nombre}"
        tablas[nombre].write.mode("overwrite").parquet(ruta)
        _log(f"fuente {nombre} escrita en {ruta}", t0)
    return cargar_fuentes(spark, cfg)


def cargar_fuentes(spark: SparkSession, cfg: Config) -> dict[str, DataFrame]:
    tablas = {n: spark.read.parquet(f"{cfg.ruta_salida}/fuentes/{n}") for n in FUENTES}
    for n in ("afiliados", "empleadores", "cotizaciones", "licencias", "afc_eventos", "macro"):
        tablas[n] = tablas[n].cache()
    return tablas


def cargar_o_generar(spark: SparkSession, cfg: Config, regenerar: bool, t0: float) -> dict[str, DataFrame]:
    if not regenerar and os.path.exists(f"{cfg.ruta_salida}/fuentes/cotizaciones"):
        _log("cargando fuentes existentes", t0)
        return cargar_fuentes(spark, cfg)
    return generar_fuentes(spark, cfg, t0)


# --------------------------------------------------------------- features
def snapshots_backtest(cfg: Config) -> tuple[list[int], list[int]]:
    """(snapshots de entrenamiento, snapshots de test) para el backtest."""
    ultimo = cfg.periodo_snapshot
    test = [add_months(ultimo, -k) for k in range(cfg.n_snapshots_test - 1, -1, -1)]
    fin_train = add_months(test[0], -cfg.gap_entrenamiento)
    train = [add_months(fin_train, -k) for k in range(cfg.n_snapshots_entrenamiento - 1, -1, -1)]
    minimo = add_months(cfg.periodo_inicio, 12 + cfg.meses_desfase)   # 12 meses de historia
    train = [s for s in train if s >= minimo]
    return train, test


def features_por_snapshot(spark: SparkSession, cfg: Config, tablas: dict[str, DataFrame],
                          snapshots: list[int], t0: float) -> DataFrame:
    """Calcula (o lee) las features de cada snapshot y devuelve la union."""
    partes = []
    for s in snapshots:
        ruta = f"{cfg.ruta_salida}/features/periodo_snapshot={s}"
        if not os.path.exists(ruta):
            f = construir_features(spark, cfg, tablas, s)
            f.drop("periodo_snapshot").write.mode("overwrite").parquet(ruta)
            f.unpersist()
            _log(f"features snapshot {s} calculadas", t0)
        partes.append(spark.read.parquet(ruta).withColumn("periodo_snapshot", F.lit(s)))
    out = partes[0]
    for p in partes[1:]:
        out = out.unionByName(p)
    return out


def dataset_entrenamiento(cfg: Config, feats: DataFrame, tablas: dict[str, DataFrame],
                          snapshot_entrenamiento: int) -> DataFrame:
    """Une features con la verdad CONOCIDA al momento de entrenar (sin mirar el futuro)."""
    verdad = rim_conocida(tablas["cotizaciones"], tablas["macro"], add_months(snapshot_entrenamiento, -1))
    return (features_para_entrenar(feats.filter(F.col("target_ya_conocido") == 0), verdad)
            .filter(F.col("periodo_target") <= add_months(snapshot_entrenamiento, -cfg.meses_desfase)))


# ---------------------------------------------------------------- backtest
def backtest(spark: SparkSession, cfg: Config, tablas: dict[str, DataFrame], t0: float) -> dict:
    train_s, test_s = snapshots_backtest(cfg)
    _log(f"backtest: entrenar con snapshots {train_s[0]}..{train_s[-1]} ({len(train_s)}), "
         f"evaluar en {test_s}", t0)
    f_train = features_por_snapshot(spark, cfg, tablas, train_s, t0)
    f_test = features_por_snapshot(spark, cfg, tablas, test_s, t0)

    ds_train = dataset_entrenamiento(cfg, f_train, tablas, test_s[0]).cache()
    _log(f"dataset entrenamiento: {ds_train.count()} filas", t0)
    modelo = HurdleModel.fit(ds_train, cfg)
    _log("modelo backtest entrenado", t0)

    verdad_final = rim_conocida(tablas["cotizaciones"], tablas["macro"], None)
    ds_test = features_para_entrenar(f_test.filter(F.col("target_ya_conocido") == 0), verdad_final)
    pred = agregar_baselines(modelo.transform(ds_test)).withColumn("tramo_renta", tramo_renta()).cache()
    _log(f"dataset test: {pred.count()} filas", t0)

    res = {
        "metricas": metricas(pred),
        "metricas_clasificador": metricas_clasificador(pred),
        "metricas_por_tipo": metricas(pred, ["tipo_afiliado"]),
        "metricas_por_tramo": metricas(pred, ["tramo_renta"]),
        "metricas_por_snapshot": metricas(pred, ["periodo_snapshot"]),
        "importancias": modelo.importancias(),
    }
    ds_train.unpersist()
    pred.unpersist()
    return res


# ---------------------------------------------------------- entrenar final
def entrenar_final(spark: SparkSession, cfg: Config, tablas: dict[str, DataFrame], t0: float) -> HurdleModel:
    fin = add_months(cfg.periodo_snapshot, -cfg.gap_entrenamiento)
    snaps = [add_months(fin, -k) for k in range(cfg.n_snapshots_entrenamiento - 1, -1, -1)]
    snaps = [s for s in snaps if s >= add_months(cfg.periodo_inicio, 12 + cfg.meses_desfase)]
    feats = features_por_snapshot(spark, cfg, tablas, snaps, t0)
    ds = dataset_entrenamiento(cfg, feats, tablas, cfg.periodo_snapshot).cache()
    _log(f"modelo final: {ds.count()} filas de {len(snaps)} snapshots", t0)
    modelo = HurdleModel.fit(ds, cfg)
    modelo.save(f"{cfg.ruta_salida}/modelo")
    ds.unpersist()
    _log(f"modelo final guardado en {cfg.ruta_salida}/modelo", t0)
    return modelo


# -------------------------------------------------------------------- demo
def _imprimir(titulo: str, df: pd.DataFrame) -> None:
    print(f"\n=== {titulo} ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:,.3f}".format):
        print(df.to_string(index=False))


def demo(cfg: Config, regenerar: bool = True) -> dict:
    t0 = time.time()
    spark = get_spark(shuffle_partitions=cfg.shuffle_partitions)
    os.makedirs(f"{cfg.ruta_salida}/reportes", exist_ok=True)
    tablas = cargar_o_generar(spark, cfg, regenerar, t0)
    hoy = dt.date.today().isoformat()

    # 1) backtest
    res = backtest(spark, cfg, tablas, t0)
    for k, v in res.items():
        v.to_csv(f"{cfg.ruta_salida}/reportes/{k}.csv", index=False)
    _imprimir("Backtest: modelo vs baselines (por horizonte)", res["metricas"])
    _imprimir("Backtest: clasificador cotiza / no cotiza", res["metricas_clasificador"])
    _imprimir("Backtest: por tipo de afiliado", res["metricas_por_tipo"][res["metricas_por_tipo"].predictor.isin(["rim_proyectada", "persistencia"])])
    _imprimir("Backtest: por tramo de renta", res["metricas_por_tramo"][res["metricas_por_tramo"].predictor.isin(["rim_proyectada", "persistencia"])])
    _imprimir("Importancia de features (top 25)", res["importancias"])

    # 2) modelo final + proyeccion de los meses abiertos
    modelo = entrenar_final(spark, cfg, tablas, t0)
    proy = proyectar(cfg, tablas, modelo, cfg.periodo_snapshot, hoy).cache()
    ruta_proy = f"{cfg.ruta_salida}/rim_proyectada"
    proy.write.mode("overwrite").parquet(ruta_proy)
    resumen = (proy.groupBy("periodo", "origen", "detalle_origen")
               .agg(F.count("*").alias("n"), F.round(F.avg("rim_valor")).alias("rim_prom"),
                    F.round(F.sum("rim_esperada") / 1e6, 1).alias("masa_esperada_MM"))
               .orderBy("periodo", "origen").toPandas())
    _imprimir(f"Proyeccion en snapshot {cfg.periodo_snapshot} (corte {cfg.periodo_corte})", resumen)
    resumen.to_csv(f"{cfg.ruta_salida}/reportes/proyeccion_resumen.csv", index=False)

    # 3) ciclo mensual siguiente: llegan los pagos de corte+1 -> reconciliar y re-proyectar
    s2 = add_months(cfg.periodo_snapshot, 1)
    existentes = spark.read.parquet(ruta_proy)
    reconciliadas = reconciliar(existentes, tablas["cotizaciones"], tablas["macro"], s2, cfg.meses_desfase)
    cfg2 = Config(**{**cfg.__dict__, "periodo_snapshot": s2})
    nuevas = proyectar(cfg2, tablas, modelo, s2, hoy)
    actualizadas = actualizar_proyecciones(reconciliadas, nuevas).cache()
    actualizadas.write.mode("overwrite").parquet(f"{cfg.ruta_salida}/rim_proyectada_{s2}")
    rep = reporte_reemplazos(actualizadas, s2)
    _imprimir(f"Ciclo {s2}: error del modelo en filas reemplazadas por el dato real", rep)
    rep.to_csv(f"{cfg.ruta_salida}/reportes/reemplazos_{s2}.csv", index=False)
    estado = (actualizadas.groupBy("periodo", "origen", "detalle_origen").count()
              .orderBy("periodo", "origen", "detalle_origen").toPandas())
    _imprimir(f"Estado de la tabla rim_proyectada tras el ciclo {s2}", estado)
    ejemplo = (actualizadas.filter("rim_predicha_previa is not null")
               .select("afiliado_id", "periodo", "origen", "detalle_origen", "rim_predicha_previa", "rim_valor",
                       "prob_cotiza", "periodo_snapshot", "periodo_reemplazo")
               .orderBy("afiliado_id", "periodo").limit(12).toPandas())
    _imprimir("Ejemplos de reemplazo PREDICHA -> REAL", ejemplo)

    salida = {"metricas": res["metricas"].to_dict(orient="records"),
              "clasificador": res["metricas_clasificador"].to_dict(orient="records"),
              "reemplazos": rep.to_dict(orient="records")}
    with open(f"{cfg.ruta_salida}/reportes/resumen.json", "w") as fh:
        json.dump(salida, fh, indent=1, default=str)
    _log("demo terminada", t0)
    spark.stop()
    return salida


# --------------------------------------------------------------------- CLI
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nowcasting de RIM (AFP) con PySpark")
    sub = p.add_subparsers(dest="cmd", required=True)
    for nombre in ("demo", "generar", "backtest", "entrenar", "proyectar"):
        sp = sub.add_parser(nombre)
        sp.add_argument("--n-afiliados", type=int, default=Config.n_afiliados)
        sp.add_argument("--salida", default=Config.ruta_salida)
        sp.add_argument("--snapshot", type=int, default=Config.periodo_snapshot)
        sp.add_argument("--inicio", type=int, default=Config.periodo_inicio)
        sp.add_argument("--seed", type=int, default=Config.seed)
        sp.add_argument("--n-train", type=int, default=Config.n_snapshots_entrenamiento)
        sp.add_argument("--n-test", type=int, default=Config.n_snapshots_test)
        sp.add_argument("--gbt-iter", type=int, default=Config.gbt_max_iter)
        sp.add_argument("--reusar-fuentes", action="store_true", help="no regenerar datos si existen")
    return p


def _cfg_desde_args(a: argparse.Namespace) -> Config:
    return Config(n_afiliados=a.n_afiliados, ruta_salida=a.salida, periodo_snapshot=a.snapshot,
                  periodo_inicio=a.inicio, seed=a.seed, n_snapshots_entrenamiento=a.n_train,
                  n_snapshots_test=a.n_test, gbt_max_iter=a.gbt_iter)


def main(argv: list[str] | None = None) -> None:
    a = _parser().parse_args(argv)
    cfg = _cfg_desde_args(a)
    if a.cmd == "demo":
        demo(cfg, regenerar=not a.reusar_fuentes)
        return
    t0 = time.time()
    spark = get_spark(shuffle_partitions=cfg.shuffle_partitions)
    if a.cmd == "generar":
        generar_fuentes(spark, cfg, t0)
    elif a.cmd == "backtest":
        tablas = cargar_o_generar(spark, cfg, not a.reusar_fuentes, t0)
        res = backtest(spark, cfg, tablas, t0)
        _imprimir("Backtest", res["metricas"])
        _imprimir("Clasificador", res["metricas_clasificador"])
    elif a.cmd == "entrenar":
        tablas = cargar_o_generar(spark, cfg, not a.reusar_fuentes, t0)
        entrenar_final(spark, cfg, tablas, t0)
    elif a.cmd == "proyectar":
        tablas = cargar_o_generar(spark, cfg, not a.reusar_fuentes, t0)
        modelo = HurdleModel.load(f"{cfg.ruta_salida}/modelo", cfg)
        proy = proyectar(cfg, tablas, modelo, cfg.periodo_snapshot, dt.date.today().isoformat())
        proy.write.mode("overwrite").parquet(f"{cfg.ruta_salida}/rim_proyectada")
        _imprimir("Proyeccion", proy.groupBy("periodo", "origen").count().orderBy("periodo").toPandas())
    spark.stop()


if __name__ == "__main__":
    main()
