"""Utilidades de periodos (yyyymm) y tabla macro sintetica (UF, IMM, tope, IPC, desempleo).

Los valores macro son APROXIMACIONES con fines de simulacion; en produccion
se reemplazan por las series oficiales (Banco Central / INE / SP).
"""
from __future__ import annotations

import math

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F


# ---------------------------------------------------------------- periodos
def periodo_to_idx(p: int) -> int:
    """yyyymm -> meses transcurridos desde 2000-01."""
    return (p // 100 - 2000) * 12 + (p % 100 - 1)


def idx_to_periodo(i: int) -> int:
    return (2000 + i // 12) * 100 + (i % 12 + 1)


def add_months(p: int, k: int) -> int:
    return idx_to_periodo(periodo_to_idx(p) + k)


def diff_months(p_fin: int, p_ini: int) -> int:
    return periodo_to_idx(p_fin) - periodo_to_idx(p_ini)


def rango_periodos(p_ini: int, p_fin: int) -> list[int]:
    return [idx_to_periodo(i) for i in range(periodo_to_idx(p_ini), periodo_to_idx(p_fin) + 1)]


# Equivalentes en expresiones Spark (evitan UDFs) ---------------------------
def col_periodo_to_idx(c: Column) -> Column:
    return (F.floor(c / 100) - 2000) * 12 + (c % 100 - 1)


def col_idx_to_periodo(c: Column) -> Column:
    return (2000 + F.floor(c / 12)) * 100 + (c % 12 + 1)


def col_add_months(c: Column, k) -> Column:
    return col_idx_to_periodo(col_periodo_to_idx(c) + k)


def col_mes(c: Column) -> Column:
    return c % 100


# ------------------------------------------------------------------- macro
_UF_ANCLAS = {202201: 31_000, 202301: 35_100, 202401: 36_800, 202501: 38_400,
              202601: 39_600, 202701: 40_800}
# Ingreso minimo mensual (aprox. historico + supuesto hacia adelante)
_IMM_ANCLAS = [(202201, 350_000), (202205, 380_000), (202208, 400_000), (202301, 410_000),
               (202305, 440_000), (202309, 460_000), (202407, 500_000), (202501, 510_636), (202505, 529_000),
               (202601, 539_000), (202605, 560_000)]
# Tope imponible en UF (se reajusta cada enero)
_TOPE_UF = {2022: 81.6, 2023: 81.6, 2024: 84.3, 2025: 87.8, 2026: 89.9, 2027: 92.0}
# IPC interanual (%) anclas
_IPC_ANCLAS = {202201: 7.7, 202207: 13.1, 202301: 12.3, 202307: 6.5, 202401: 3.8, 202407: 4.6,
               202501: 4.9, 202507: 4.3, 202601: 3.6, 202607: 3.4, 202701: 3.2}
# Desempleo INE (%) base y estacionalidad (peak Mar-May, valle Dic)
_DESEMPLEO_BASE = {2022: 7.9, 2023: 8.7, 2024: 8.5, 2025: 8.3, 2026: 8.0, 2027: 7.8}
_DESEMPLEO_ESTACIONAL = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.7, 5: 0.6, 6: 0.3,
                         7: 0.1, 8: -0.1, 9: -0.2, 10: -0.3, 11: -0.4, 12: -0.5}


def _interp_anclas(anclas: dict[int, float], p: int, geometrico: bool) -> float:
    idx = periodo_to_idx(p)
    pts = sorted((periodo_to_idx(k), v) for k, v in anclas.items())
    if idx <= pts[0][0]:
        return pts[0][1]
    if idx >= pts[-1][0]:
        return pts[-1][1]
    for (i0, v0), (i1, v1) in zip(pts, pts[1:]):
        if i0 <= idx <= i1:
            w = (idx - i0) / (i1 - i0)
            if geometrico:
                return math.exp(math.log(v0) * (1 - w) + math.log(v1) * w)
            return v0 * (1 - w) + v1 * w
    raise ValueError(p)


def _imm(p: int) -> int:
    val = _IMM_ANCLAS[0][1]
    for per, v in _IMM_ANCLAS:
        if per <= p:
            val = v
    return val


def macro_pandas(p_ini: int, p_fin: int) -> pd.DataFrame:
    filas = []
    for p in rango_periodos(p_ini, p_fin):
        anio, mes = p // 100, p % 100
        uf = _interp_anclas(_UF_ANCLAS, p, geometrico=True)
        tope_uf = _TOPE_UF.get(anio, _TOPE_UF[max(_TOPE_UF)])
        filas.append({
            "periodo": p,
            "mes": mes,
            "uf": round(uf, 2),
            "imm": _imm(p),
            "tope_uf": tope_uf,
            "tope_clp": round(tope_uf * uf, 0),
            "ipc_12m": round(_interp_anclas(_IPC_ANCLAS, p, geometrico=False), 2),
            "desempleo": round(_DESEMPLEO_BASE.get(anio, 8.0) + _DESEMPLEO_ESTACIONAL[mes], 2),
        })
    df = pd.DataFrame(filas)
    # crecimiento del IMM respecto de 12 meses atras (reajustes legales)
    df["imm_var_12m"] = (df["imm"] / df["imm"].shift(12) - 1).fillna(0.0).round(4)
    return df


def macro_spark(spark: SparkSession, p_ini: int, p_fin: int) -> DataFrame:
    return spark.createDataFrame(macro_pandas(p_ini, p_fin))
