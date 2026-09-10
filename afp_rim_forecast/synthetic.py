"""Generador de datos sinteticos para una AFP (afiliados, empleadores, verdad mensual,
movimientos de cotizacion con desfase de pago, licencias y eventos AFC).

Estrategia de escala: las dimensiones se generan con expresiones Spark
(spark.range + rand) y la dinamica mensual se simula por particion con
`mapInPandas` (numpy vectorizado sobre los afiliados de la particion, loop
sobre meses). No hay cross joins ni colecciones al driver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .calendario import (col_add_months, macro_pandas, periodo_to_idx, rango_periodos)
from .config import Config

# ------------------------------------------------------------------ rubros
@dataclass(frozen=True)
class Rubro:
    codigo: str
    peso_empleadores: float
    factor_salarial: float
    sigma_variable: float     # remuneracion variable (comisiones, turnos, bonos) como fraccion del sueldo
    p_horas_extra: float
    p_sep_base: float         # prob. mensual de termino de contrato
    p_plazo_fijo: float       # prob. de contrato plazo fijo / por obra
    mes_bono: int = 0         # bono anual (mineria/finanzas)
    p_bono: float = 0.0
    aguinaldo_medio: float = 60_000


RUBROS: list[Rubro] = [
    Rubro("A_AGRICULTURA", 0.10, 0.75, 0.15, 0.30, 0.10, 0.70, aguinaldo_medio=30_000),
    Rubro("B_MINERIA", 0.03, 2.20, 0.25, 0.50, 0.02, 0.20, mes_bono=3, p_bono=0.7,
          aguinaldo_medio=250_000),
    Rubro("C_MANUFACTURA", 0.10, 1.00, 0.10, 0.40, 0.035, 0.30, aguinaldo_medio=70_000),
    Rubro("F_CONSTRUCCION", 0.12, 0.95, 0.20, 0.45, 0.09, 0.60, aguinaldo_medio=40_000),
    Rubro("G_COMERCIO", 0.18, 0.85, 0.20, 0.35, 0.06, 0.40, aguinaldo_medio=50_000),
    Rubro("H_TRANSPORTE", 0.06, 0.95, 0.15, 0.50, 0.045, 0.30, aguinaldo_medio=50_000),
    Rubro("I_TURISMO", 0.05, 0.75, 0.15, 0.30, 0.08, 0.60, aguinaldo_medio=30_000),
    Rubro("K_FINANZAS", 0.04, 1.80, 0.15, 0.10, 0.02, 0.10, mes_bono=3, p_bono=0.8,
          aguinaldo_medio=150_000),
    Rubro("N_SERVICIOS", 0.10, 0.80, 0.10, 0.30, 0.06, 0.50, aguinaldo_medio=35_000),
    Rubro("O_ADMIN_PUBLICA", 0.07, 1.20, 0.03, 0.10, 0.012, 0.25, aguinaldo_medio=90_000),
    Rubro("P_EDUCACION", 0.07, 1.00, 0.05, 0.10, 0.03, 0.40, aguinaldo_medio=60_000),
    Rubro("Q_SALUD", 0.05, 1.15, 0.12, 0.40, 0.025, 0.30, aguinaldo_medio=70_000),
    Rubro("T_CASA_PARTICULAR", 0.03, 0.65, 0.05, 0.05, 0.05, 0.10, aguinaldo_medio=20_000),
]

# Estacionalidad explicita (terminos / contrataciones) por rubro
_SEASON_SEP = {
    "A_AGRICULTURA": {3: 1.5, 4: 2.5, 5: 2.5, 6: 1.3},
    "F_CONSTRUCCION": {6: 1.3, 7: 1.3},
    "G_COMERCIO": {1: 1.8, 2: 1.5},
    "I_TURISMO": {3: 2.0, 4: 1.5},
    "P_EDUCACION": {12: 1.5, 1: 3.0, 2: 3.0},
}
_SEASON_HIRE = {
    "A_AGRICULTURA": {11: 2.5, 12: 2.5, 1: 2.0, 2: 1.5},
    "G_COMERCIO": {11: 2.0, 12: 2.0},
    "I_TURISMO": {12: 2.0, 1: 2.0},
    "P_EDUCACION": {3: 3.0, 2: 1.5},
    "F_CONSTRUCCION": {9: 1.3, 10: 1.3},
}
RUBRO_IDX = {r.codigo: i for i, r in enumerate(RUBROS)}
N_RUBROS = len(RUBROS)


def _season_matrix(tabla: dict[str, dict[int, float]]) -> np.ndarray:
    m = np.ones((N_RUBROS, 13))
    for cod, meses in tabla.items():
        for mes, v in meses.items():
            m[RUBRO_IDX[cod], mes] = v
    return m


SEASON_SEP = _season_matrix(_SEASON_SEP)
SEASON_HIRE = _season_matrix(_SEASON_HIRE)

TAMANOS = ["MICRO", "PEQUENA", "MEDIANA", "GRANDE"]
TAMANO_P = [0.45, 0.35, 0.15, 0.05]
TAMANO_PESO_CONTRATACION = {"MICRO": 1.0, "PEQUENA": 4.0, "MEDIANA": 15.0, "GRANDE": 60.0}
# comportamiento de pago por tamano: (anticipado, oportuno m+1, tardio m+2, rezago 3..8, dnp, rectificacion)
TAMANO_PAGO = {
    "MICRO":   (0.01, 0.84, 0.07, 0.08, 0.06, 0.02),
    "PEQUENA": (0.02, 0.88, 0.05, 0.05, 0.04, 0.015),
    "MEDIANA": (0.04, 0.92, 0.03, 0.01, 0.02, 0.01),
    "GRANDE":  (0.06, 0.93, 0.01, 0.00, 0.005, 0.01),
}


# ------------------------------------------------------------- empleadores
def generar_empleadores_pandas(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    n = cfg.n_empleadores
    rubro_idx = rng.choice(N_RUBROS, size=n, p=[r.peso_empleadores for r in RUBROS])
    tamano = rng.choice(TAMANOS, size=n, p=TAMANO_P)
    region = rng.choice(np.arange(1, 17), size=n,
                        p=np.array([2, 2, 2, 3, 8, 3, 4, 7, 3, 5, 1, 1, 42, 2, 1, 14]) / 100)
    politica_grat = rng.choice(["MENSUAL", "ANUAL_ABRIL", "NINGUNA"], size=n, p=[0.70, 0.20, 0.10])
    freq_reajuste = rng.choice(["ANUAL_ENE", "SEMESTRAL", "ANUAL_DIC", "NINGUNO"], size=n,
                               p=[0.45, 0.20, 0.05, 0.30])
    filas = []
    for i in range(n):
        r = RUBROS[rubro_idx[i]]
        t = tamano[i]
        pa, po, pt, pr, pdnp, prect = TAMANO_PAGO[t]
        publico = r.codigo == "O_ADMIN_PUBLICA"
        filas.append({
            "empleador_id": i + 1,
            "rubro": r.codigo,
            "tamano_empleador": t,
            "region_empleador": int(region[i]),
            "politica_gratificacion": "MENSUAL" if publico else politica_grat[i],
            "freq_reajuste": "ANUAL_DIC" if publico else freq_reajuste[i],
            "aguinaldo_sep": float(rng.gamma(2.0, r.aguinaldo_medio / 2) * (rng.random() < 0.7)),
            "aguinaldo_dic": float(rng.gamma(2.0, r.aguinaldo_medio / 2) * (rng.random() < 0.8)),
            "factor_salarial_emp": float(np.exp(rng.normal(0, 0.15))),
            "p_pago_anticipado": 0.02 if publico else pa,
            "p_pago_oportuno": 0.97 if publico else po,
            "p_pago_tardio": 0.01 if publico else pt,
            "p_rezago": 0.0 if publico else pr,
            "p_dnp": 0.0 if publico else pdnp,
            "p_rectificacion": prect,
            "peso_contratacion": TAMANO_PESO_CONTRATACION[t] * (3.0 if publico else 1.0),
        })
    return pd.DataFrame(filas)


def generar_empleadores(spark: SparkSession, cfg: Config) -> DataFrame:
    return spark.createDataFrame(generar_empleadores_pandas(cfg))


# ---------------------------------------------------------------- afiliados
def generar_afiliados(spark: SparkSession, cfg: Config) -> DataFrame:
    """Dimension de afiliados con atributos observables y latentes (prefijo lat_)."""
    seed = cfg.seed
    n_nuevos = 0.10   # afiliados que se incorporan durante la ventana simulada (cold start)
    idx_ini = periodo_to_idx(cfg.periodo_inicio)
    idx_fin = periodo_to_idx(cfg.periodo_snapshot)
    r = lambda k: F.rand(seed + k)  # noqa: E731
    df = (
        spark.range(1, cfg.n_afiliados + 1).withColumnRenamed("id", "afiliado_id")
        .withColumn("sexo", F.when(r(11) < 0.47, "F").otherwise("M"))
        .withColumn("edad_inicio", (18 + F.floor((r(12) + r(13)) / 2 * 47)).cast("int"))
        .withColumn("region", F.when(r(14) < 0.42, 13).when(r(14) < 0.52, 5).when(r(14) < 0.62, 8)
                    .otherwise((1 + F.floor(r(15) * 16)).cast("int")))
        .withColumn("nivel_educacional", F.when(r(16) < 0.15, "BASICA").when(r(16) < 0.60, "MEDIA")
                    .when(r(16) < 0.80, "TECNICA").otherwise("UNIVERSITARIA"))
        .withColumn("tipo_afiliado", F.when((F.col("edad_inicio") >= 60) & (r(17) < 0.5), "PENSIONADO_ACTIVO")
                    .when(r(17) < 0.08, "INDEPENDIENTE").when(r(17) < 0.10, "VOLUNTARIO").otherwise("DEPENDIENTE"))
        .withColumn("_idx_afil", F.when(r(18) < n_nuevos, idx_ini + 1 + F.floor(r(19) * (idx_fin - idx_ini - 1)))
                    .otherwise(idx_ini - F.floor(r(19) * 240)))
        .withColumn("periodo_afiliacion", ((2000 + F.floor(F.col("_idx_afil") / 12)) * 100
                                           + (F.col("_idx_afil") % 12 + 1)).cast("int"))
        .drop("_idx_afil")
        # latentes (no observables por el modelo; se usan solo para simular)
        .withColumn("lat_productividad", F.exp(F.randn(seed + 20) * 0.45
                    + F.when(F.col("nivel_educacional") == "UNIVERSITARIA", 0.55)
                    .when(F.col("nivel_educacional") == "TECNICA", 0.25)
                    .when(F.col("nivel_educacional") == "BASICA", -0.20).otherwise(0.0)))
        .withColumn("lat_rubro_pref", F.floor(r(21) * N_RUBROS).cast("int"))
        .withColumn("lat_p_sep_factor", F.exp(F.randn(seed + 22) * 0.4))
        .withColumn("lat_p_lic_factor", F.exp(F.randn(seed + 23) * 0.5))
        .withColumn("lat_multi_empleador", (r(24) < 0.05).cast("int"))
        .withColumn("lat_jornada", F.when(r(25) < 0.12, 0.5).otherwise(1.0))
        .withColumn("lat_masa_imm", (r(26) < 0.15).cast("int"))
    )
    return df


# --------------------------------------------------------- verdad mensual
VERDAD_SCHEMA = T.StructType([
    T.StructField("afiliado_id", T.LongType()),
    T.StructField("periodo", T.IntegerType()),
    T.StructField("estado", T.StringType()),
    T.StructField("empleador_id", T.LongType(), True),
    T.StructField("empleador2_id", T.LongType(), True),
    T.StructField("tipo_contrato", T.StringType(), True),
    T.StructField("jornada", T.DoubleType()),
    T.StructField("sueldo_base", T.DoubleType()),
    T.StructField("rim_emp1", T.DoubleType()),
    T.StructField("rim_emp2", T.DoubleType()),
    T.StructField("gratificacion", T.DoubleType()),
    T.StructField("aguinaldo", T.DoubleType()),
    T.StructField("variable", T.DoubleType()),
    T.StructField("dias_licencia", T.IntegerType()),
    T.StructField("evento_afc", T.StringType(), True),
    T.StructField("meses_antiguedad", T.IntegerType()),
    T.StructField("edad", T.IntegerType()),
])


def _simular_particion(emp: pd.DataFrame, macro: pd.DataFrame, periodos: list[int], seed: int):
    """Devuelve la funcion que simula un lote de afiliados (uso en mapInPandas)."""
    # arrays de empleadores
    emp = emp.sort_values("empleador_id").reset_index(drop=True)
    e_id = emp["empleador_id"].to_numpy()
    e_rubro = emp["rubro"].map(RUBRO_IDX).to_numpy()
    e_grat = emp["politica_gratificacion"].to_numpy()
    e_reaj = emp["freq_reajuste"].to_numpy()
    e_ag_sep = emp["aguinaldo_sep"].to_numpy()
    e_ag_dic = emp["aguinaldo_dic"].to_numpy()
    e_factor = emp["factor_salarial_emp"].to_numpy()
    e_peso = emp["peso_contratacion"].to_numpy()
    r_factor = np.array([r.factor_salarial for r in RUBROS])
    r_sigma = np.array([r.sigma_variable for r in RUBROS])
    r_phe = np.array([r.p_horas_extra for r in RUBROS])
    r_psep = np.array([r.p_sep_base for r in RUBROS])
    r_ppf = np.array([r.p_plazo_fijo for r in RUBROS])
    r_mes_bono = np.array([r.mes_bono for r in RUBROS])
    r_pbono = np.array([r.p_bono for r in RUBROS])
    # muestreo de empleadores por rubro (ponderado por tamano)
    por_rubro = []
    for k in range(N_RUBROS):
        ix = np.where(e_rubro == k)[0]
        if len(ix) == 0:
            ix = np.arange(len(e_id))
        w = e_peso[ix]
        por_rubro.append((ix, np.cumsum(w) / w.sum()))

    macro = macro.set_index("periodo")
    imm_v = macro.loc[periodos, "imm"].to_numpy(dtype=float)
    tope_v = macro.loc[periodos, "tope_clp"].to_numpy(dtype=float)
    ipc_v = macro.loc[periodos, "ipc_12m"].to_numpy(dtype=float) / 100.0
    des_v = macro.loc[periodos, "desempleo"].to_numpy(dtype=float) / 100.0
    imm0 = imm_v[0]
    p_idx = np.array([periodo_to_idx(p) for p in periodos])
    meses = np.array([p % 100 for p in periodos])

    def elegir_empleador(rng, rubro_pref, n):
        out = np.empty(n, dtype=np.int64)
        usa_pref = rng.random(n) < 0.7
        rubro = np.where(usa_pref, rubro_pref, rng.integers(0, N_RUBROS, n))
        for k in range(N_RUBROS):
            m = rubro == k
            if m.any():
                ix, cw = por_rubro[k]
                out[m] = ix[np.searchsorted(cw, rng.random(m.sum()))]
        return out

    def sueldo_nuevo(rng, prod, rubro_idx_emp, factor_emp, jornada, masa_imm, imm, sueldo_prev):
        base_mediana = 600_000 * (imm / imm0) ** 0.9
        fresco = prod * r_factor[rubro_idx_emp] * factor_emp * base_mediana * np.exp(rng.normal(0, 0.25, len(prod)))
        recontr = sueldo_prev * np.exp(rng.normal(0.03, 0.12, len(prod)))
        s = np.where((sueldo_prev > 0) & (rng.random(len(prod)) < 0.4), recontr, fresco)
        s = np.where(masa_imm == 1, imm, s)
        return np.maximum(s, imm) * jornada

    def simular(iterator):
        for pdf in iterator:
            n = len(pdf)
            if n == 0:
                continue
            rng = np.random.default_rng(seed * 1_000_003 + int(pdf["afiliado_id"].iloc[0]))
            af_id = pdf["afiliado_id"].to_numpy()
            tipo = pdf["tipo_afiliado"].to_numpy()
            sexo = pdf["sexo"].to_numpy()
            edad = pdf["edad_inicio"].to_numpy().astype(int)
            prod = pdf["lat_productividad"].to_numpy(dtype=float)
            rubro_pref = pdf["lat_rubro_pref"].to_numpy().astype(int)
            f_sep = pdf["lat_p_sep_factor"].to_numpy(dtype=float)
            f_lic = pdf["lat_p_lic_factor"].to_numpy(dtype=float)
            multi = pdf["lat_multi_empleador"].to_numpy().astype(int)
            jornada = pdf["lat_jornada"].to_numpy(dtype=float)
            masa_imm = pdf["lat_masa_imm"].to_numpy().astype(int)
            afil_idx = np.array([periodo_to_idx(int(p)) for p in pdf["periodo_afiliacion"]])

            dependiente = np.isin(tipo, ["DEPENDIENTE", "PENSIONADO_ACTIVO"])
            independiente = tipo == "INDEPENDIENTE"
            voluntario = tipo == "VOLUNTARIO"

            # estado inicial
            emp = np.full(n, -1, dtype=np.int64)          # indice en arrays de empleadores
            sueldo = np.zeros(n)
            sueldo_prev = np.zeros(n)
            tenure = np.zeros(n, dtype=int)
            plazo_fijo = np.zeros(n, dtype=bool)
            lic_rest = np.zeros(n, dtype=int)
            retirado = np.zeros(n, dtype=bool)
            emp2 = np.full(n, -1, dtype=np.int64)
            sueldo2 = np.zeros(n)
            renta_indep_anual = np.zeros(n)
            activo_inicial = dependiente & (rng.random(n) < 0.78)
            k0 = activo_inicial.sum()
            if k0:
                emp[activo_inicial] = elegir_empleador(rng, rubro_pref[activo_inicial], k0)
                sueldo[activo_inicial] = sueldo_nuevo(
                    rng, prod[activo_inicial], e_rubro[emp[activo_inicial]], e_factor[emp[activo_inicial]],
                    jornada[activo_inicial], masa_imm[activo_inicial], imm_v[0], np.zeros(k0))
                tenure[activo_inicial] = rng.integers(1, 120, k0)
                plazo_fijo[activo_inicial] = rng.random(k0) < r_ppf[e_rubro[emp[activo_inicial]]]
            filas = []
            for t, periodo in enumerate(periodos):
                mes = int(meses[t])
                imm, tope, ipc, des = imm_v[t], tope_v[t], ipc_v[t], des_v[t]
                if mes == 1 and t > 0:
                    edad = edad + 1
                    # renta anual de independientes se redefine cada anio
                afiliado_ya = afil_idx <= p_idx[t]
                recien_afiliado = afil_idx == p_idx[t]
                evento = np.full(n, None, dtype=object)

                # --- retiro / jubilacion
                edad_ret = np.where(sexo == "F", 60, 65)
                se_retira = dependiente & ~retirado & (edad >= edad_ret) & (rng.random(n) < 0.04)
                retirado |= se_retira

                # --- terminos de contrato
                empleado = emp >= 0
                rubro_e = np.where(empleado, e_rubro[np.maximum(emp, 0)], 0)
                p_sep = r_psep[rubro_e] * SEASON_SEP[rubro_e, mes] * f_sep * np.where(plazo_fijo, 1.8, 1.0) \
                    * np.where(tenure > 24, 0.5, 1.0) * np.where(tipo == "PENSIONADO_ACTIVO", 1.5, 1.0)
                termina = empleado & ((rng.random(n) < p_sep) | se_retira)
                fraccion_mes = np.ones(n)
                fraccion_mes[termina] = rng.uniform(0.2, 1.0, termina.sum())
                evento[termina] = "TERMINO"

                # --- contrataciones (afiliados dependientes cesantes, no retirados)
                cesante = dependiente & ~empleado & ~retirado & afiliado_ya
                p_hire_base = 0.16 * (1 - 4 * (des - 0.08))
                p_hire = p_hire_base * SEASON_HIRE[rubro_pref, mes] * np.where(edad > 55, 0.5, 1.0)
                contrata = cesante & ((rng.random(n) < p_hire) | (recien_afiliado & (rng.random(n) < 0.9)))
                kc = contrata.sum()
                if kc:
                    nuevos = elegir_empleador(rng, rubro_pref[contrata], kc)
                    emp[contrata] = nuevos
                    sueldo[contrata] = sueldo_nuevo(rng, prod[contrata], e_rubro[nuevos], e_factor[nuevos],
                                                    jornada[contrata], masa_imm[contrata], imm, sueldo_prev[contrata])
                    tenure[contrata] = 0
                    plazo_fijo[contrata] = rng.random(kc) < r_ppf[e_rubro[nuevos]]
                    fraccion_mes[contrata] = rng.uniform(0.3, 1.0, kc)
                    evento[contrata] = "INICIO"
                empleado = emp >= 0
                rubro_e = np.where(empleado, e_rubro[np.maximum(emp, 0)], 0)
                emp_safe = np.maximum(emp, 0)

                # --- reajustes y piso legal
                freq = e_reaj[emp_safe]
                reajusta = empleado & (tenure >= 1) & (
                    ((freq == "ANUAL_ENE") & (mes == 1)) | ((freq == "ANUAL_DIC") & (mes == 12))
                    | ((freq == "SEMESTRAL") & np.isin(mes, [1, 7])))
                tasa = np.where(freq == "SEMESTRAL", ipc / 2, ipc) * rng.uniform(0.8, 1.3, n)
                sueldo = np.where(reajusta, sueldo * (1 + tasa), sueldo)
                sueldo = np.where(empleado, np.maximum(sueldo, imm * jornada), sueldo)

                # --- licencias medicas
                nueva_lic = empleado & (lic_rest == 0) & (rng.random(n) < 0.025 * f_lic)
                lic_rest = np.where(nueva_lic, 1 + rng.geometric(0.55, n), lic_rest)
                en_lic = empleado & (lic_rest > 0)
                dias_lic = np.where(en_lic, np.where(lic_rest == 1, rng.integers(5, 31, n), 30), 0)
                lic_rest = np.maximum(lic_rest - 1, 0)

                # --- componentes de la remuneracion imponible
                grat = np.where((e_grat[emp_safe] == "MENSUAL") & empleado,
                                np.minimum(0.25 * sueldo, 4.75 * imm / 12), 0.0)
                grat_anual = (e_grat[emp_safe] == "ANUAL_ABRIL") & empleado & (mes == 4)
                grat = np.where(grat_anual, np.minimum(3.0 * sueldo, 4.75 * imm) * np.minimum(tenure, 12) / 12, grat)
                aguinaldo = np.where(empleado & (mes == 9), e_ag_sep[emp_safe], 0.0) \
                    + np.where(empleado & (mes == 12), e_ag_dic[emp_safe], 0.0)
                sig = r_sigma[rubro_e]
                variable = sueldo * np.maximum(0.0, rng.normal(sig, sig, n))
                he = (rng.random(n) < r_phe[rubro_e]) * sueldo * rng.uniform(0.02, 0.15, n)
                bono = np.where((r_mes_bono[rubro_e] == mes) & (rng.random(n) < r_pbono[rubro_e]),
                                sueldo * rng.uniform(0.8, 2.0, n), 0.0)
                rim1 = np.where(empleado, (sueldo + variable + he) * fraccion_mes + grat + aguinaldo + bono, 0.0)
                rim1 = np.where(en_lic, rim1 * (1 - 0.10 * dias_lic / 30), rim1)   # subsidio ~ 90% de la base
                rim1 = np.minimum(rim1, tope)

                # --- segundo empleador (jornada parcial adicional)
                quiere2 = (multi == 1) & empleado
                nuevo2 = quiere2 & (emp2 < 0)
                k2 = nuevo2.sum()
                if k2:
                    emp2[nuevo2] = elegir_empleador(rng, rubro_pref[nuevo2], k2)
                    sueldo2[nuevo2] = np.maximum(sueldo[nuevo2] * rng.uniform(0.2, 0.6, k2), imm * 0.5)
                emp2 = np.where(quiere2, emp2, -1)
                rim2 = np.where(quiere2, np.minimum(sueldo2 * (1 + rng.uniform(0, 0.1, n)), tope), 0.0)
                sueldo2 = np.where(quiere2, sueldo2 * (1 + np.where(mes == 1, ipc, 0.0)), 0.0)

                # --- independientes: renta anual / 12 (Operacion Renta). ~30% sin renta en el anio
                if mes == 1 or t == 0:
                    renta_indep_anual = np.where(independiente & (rng.random(n) > 0.30),
                                                 prod * 9_000_000 * (imm / imm0) * np.exp(rng.normal(0, 0.5, n)), 0.0)
                rim_ind = np.where(independiente & afiliado_ya, np.minimum(renta_indep_anual / 12, tope), 0.0)
                # --- voluntarios: pagos esporadicos
                rim_vol = np.where(voluntario & afiliado_ya & (rng.random(n) < 0.25),
                                   np.minimum(imm * np.exp(rng.normal(0.2, 0.5, n)), tope), 0.0)

                estado = np.where(retirado, "RETIRADO",
                         np.where(independiente, "INDEPENDIENTE",
                         np.where(voluntario, "VOLUNTARIO",
                         np.where(en_lic, "LICENCIA", np.where(empleado, "EMPLEADO", "CESANTE")))))
                rim_emp1 = np.where(independiente, rim_ind, np.where(voluntario, rim_vol, rim1))

                filas.append(pd.DataFrame({
                    "afiliado_id": af_id,
                    "periodo": np.full(n, periodo, dtype=np.int32),
                    "estado": estado,
                    "empleador_id": pd.array(np.where(empleado, e_id[emp_safe], 0), dtype="Int64"),
                    "empleador2_id": pd.array(np.where(emp2 >= 0, e_id[np.maximum(emp2, 0)], 0), dtype="Int64"),
                    "tipo_contrato": np.where(empleado, np.where(plazo_fijo, "PLAZO_FIJO", "INDEFINIDO"), None),
                    "jornada": jornada,
                    "sueldo_base": np.where(empleado, sueldo, 0.0),
                    "rim_emp1": np.round(rim_emp1, 0),
                    "rim_emp2": np.round(rim2, 0),
                    "gratificacion": np.round(grat, 0),
                    "aguinaldo": np.round(aguinaldo, 0),
                    "variable": np.round(variable + he + bono, 0),
                    "dias_licencia": dias_lic.astype(np.int32),
                    "evento_afc": evento,
                    "meses_antiguedad": np.where(empleado, tenure, 0).astype(np.int32),
                    "edad": edad.astype(np.int32),
                })[afiliado_ya])

                # --- cierre del mes: terminos efectivos, antiguedad
                sueldo_prev = np.where(termina, sueldo, sueldo_prev)
                emp = np.where(termina, -1, emp)
                sueldo = np.where(termina, 0.0, sueldo)
                lic_rest = np.where(termina, 0, lic_rest)
                tenure = np.where(emp >= 0, tenure + 1, 0)
            out = pd.concat(filas, ignore_index=True)
            out["empleador_id"] = out["empleador_id"].where(out["empleador_id"] > 0, pd.NA)
            out["empleador2_id"] = out["empleador2_id"].where(out["empleador2_id"] > 0, pd.NA)
            yield out

    return simular


def simular_verdad_mensual(spark: SparkSession, cfg: Config, afiliados: DataFrame,
                           empleadores: DataFrame) -> DataFrame:
    """Verdad latente mes a mes por afiliado (lo que el empleador terminara declarando)."""
    periodos = rango_periodos(cfg.periodo_inicio, cfg.periodo_snapshot)
    emp_pdf = empleadores.toPandas()          # dimension pequena: viaja en el closure
    macro = macro_pandas(cfg.periodo_inicio, cfg.periodo_snapshot)
    fn = _simular_particion(emp_pdf, macro, periodos, cfg.seed)
    cols = ["afiliado_id", "tipo_afiliado", "sexo", "edad_inicio", "periodo_afiliacion",
            "lat_productividad", "lat_rubro_pref", "lat_p_sep_factor", "lat_p_lic_factor",
            "lat_multi_empleador", "lat_jornada", "lat_masa_imm"]
    return (afiliados.select(*cols)
            .repartition(cfg.particiones_simulacion, "afiliado_id")
            .sortWithinPartitions("afiliado_id")
            .mapInPandas(fn, schema=VERDAD_SCHEMA))


# ------------------------------------------------------ movimientos (pagos)
def generar_cotizaciones(cfg: Config, verdad: DataFrame, empleadores: DataFrame) -> DataFrame:
    """Movimientos de cotizacion tal como los recibe la AFP, con el mes en que llegan.

    Grano: (afiliado, periodo de remuneracion, entidad pagadora/empleador, movimiento).
    - EMPLEADOR: paga en m+1 normalmente; anticipado (m), tardio (m+2), rezago (m+3..m+8).
      Puede DECLARAR Y NO PAGAR (DNP): la RIM se conoce igual (declarada).
    - SUBSIDIO (licencia medica): la ISAPRE/FONASA cotiza por los dias de licencia con mas rezago.
    - SII: independientes, toda la renta del anio llega en junio del anio siguiente.
    - AFILIADO: voluntarios, pagan directo (m o m+1).
    Un porcentaje de declaraciones recibe una RECTIFICACION posterior.
    """
    s = cfg.seed
    emp_pago = F.broadcast(empleadores.select(
        "empleador_id", "p_pago_anticipado", "p_pago_oportuno", "p_pago_tardio", "p_rezago", "p_dnp",
        "p_rectificacion"))

    base = verdad.select("afiliado_id", "periodo", "estado", "empleador_id", "empleador2_id",
                         "rim_emp1", "rim_emp2", "dias_licencia")

    # empleador principal (parte pagada por el empleador)
    emp1 = (base.filter(F.col("empleador_id").isNotNull() & (F.col("rim_emp1") > 0))
            .withColumn("rim", F.round(F.col("rim_emp1") * (30 - F.col("dias_licencia")) / 30, 0))
            .withColumn("entidad_pagadora", F.lit("EMPLEADOR"))
            .select("afiliado_id", "periodo", "empleador_id", "entidad_pagadora", "rim", "dias_licencia"))
    # subsidio por licencia (misma remuneracion, pagador distinto, mas lento)
    subs = (base.filter(F.col("dias_licencia") > 0)
            .withColumn("rim", F.round(F.col("rim_emp1") * F.col("dias_licencia") / 30, 0))
            .withColumn("entidad_pagadora", F.lit("SUBSIDIO"))
            .withColumn("empleador_id", F.col("empleador_id"))
            .select("afiliado_id", "periodo", "empleador_id", "entidad_pagadora", "rim", "dias_licencia"))
    emp2 = (base.filter(F.col("empleador2_id").isNotNull() & (F.col("rim_emp2") > 0))
            .select("afiliado_id", "periodo", F.col("empleador2_id").alias("empleador_id"),
                    F.lit("EMPLEADOR").alias("entidad_pagadora"), F.col("rim_emp2").alias("rim"),
                    F.lit(0).alias("dias_licencia")))
    indep = (base.filter((F.col("estado") == "INDEPENDIENTE") & (F.col("rim_emp1") > 0))
             .select("afiliado_id", "periodo", F.lit(None).cast("long").alias("empleador_id"),
                     F.lit("SII").alias("entidad_pagadora"), F.col("rim_emp1").alias("rim"),
                     F.lit(0).alias("dias_licencia")))
    volun = (base.filter((F.col("estado") == "VOLUNTARIO") & (F.col("rim_emp1") > 0))
             .select("afiliado_id", "periodo", F.lit(None).cast("long").alias("empleador_id"),
                     F.lit("AFILIADO").alias("entidad_pagadora"), F.col("rim_emp1").alias("rim"),
                     F.lit(0).alias("dias_licencia")))

    mov = (emp1.unionByName(subs).unionByName(emp2).unionByName(indep).unionByName(volun)
           .filter(F.col("rim") > 0)
           .join(emp_pago, "empleador_id", "left"))

    r = F.rand(s + 100)
    lag_emp = (F.when(r < F.col("p_pago_anticipado"), 0)
               .when(r < F.col("p_pago_anticipado") + F.col("p_pago_oportuno"), 1)
               .when(r < F.col("p_pago_anticipado") + F.col("p_pago_oportuno") + F.col("p_pago_tardio"), 2)
               .otherwise(3 + F.floor(F.rand(s + 101) * 6)))
    lag = (F.when(F.col("entidad_pagadora") == "EMPLEADOR", lag_emp)
           .when(F.col("entidad_pagadora") == "SUBSIDIO", 2 + F.floor(F.rand(s + 102) * 3))
           .when(F.col("entidad_pagadora") == "AFILIADO", F.floor(F.rand(s + 103) * 2))
           .otherwise(F.lit(None)))
    recepcion = (F.when(F.col("entidad_pagadora") == "SII", (F.floor(F.col("periodo") / 100) + 1) * 100 + 6)
                 .otherwise(col_add_months(F.col("periodo"), lag)))
    mov = (mov.withColumn("periodo_recepcion", recepcion.cast("int"))
              .withColumn("estado_pago", F.when((F.col("entidad_pagadora") == "EMPLEADOR")
                                                & (F.rand(s + 104) < F.col("p_dnp")), "DNP").otherwise("PAGADA"))
              .withColumn("tipo_movimiento", F.lit("DECLARACION"))
              .withColumn("_rect", (F.col("entidad_pagadora") == "EMPLEADOR")
                          & (F.rand(s + 105) < F.col("p_rectificacion"))))

    decl = mov.select("afiliado_id", "periodo", "empleador_id", "entidad_pagadora", "tipo_movimiento",
                      "estado_pago", "rim", "periodo_recepcion", "_rect")
    rect = (mov.filter("_rect")
            .withColumn("tipo_movimiento", F.lit("RECTIFICACION"))
            .withColumn("rim", F.round(F.col("rim") * (0.85 + F.rand(s + 106) * 0.35), 0))
            .withColumn("periodo_recepcion",
                        col_add_months(F.col("periodo_recepcion"), 1 + F.floor(F.rand(s + 107) * 3)).cast("int"))
            .select(*decl.columns))
    # la regularizacion de una DNP (pago posterior) no cambia la RIM declarada: se registra como PAGO
    pago_dnp = (mov.filter("estado_pago = 'DNP'")
                .withColumn("tipo_movimiento", F.lit("PAGO_DNP"))
                .withColumn("estado_pago", F.lit("PAGADA"))
                .withColumn("periodo_recepcion",
                            col_add_months(F.col("periodo_recepcion"), 2 + F.floor(F.rand(s + 108) * 5)).cast("int"))
                .select(*decl.columns))
    out = (decl.unionByName(rect).unionByName(pago_dnp).drop("_rect")
           .withColumn("movimiento_id", F.monotonically_increasing_id()))
    return out.select("movimiento_id", "afiliado_id", "periodo", "empleador_id", "entidad_pagadora",
                      "tipo_movimiento", "estado_pago", "rim", "periodo_recepcion")


def generar_licencias(verdad: DataFrame) -> DataFrame:
    """Licencias medicas (COMPIN/ISAPRE informan mas rapido que el pago: conocidas en m+1)."""
    return (verdad.filter(F.col("dias_licencia") > 0)
            .select("afiliado_id", "periodo", "dias_licencia",
                    col_add_months(F.col("periodo"), 1).cast("int").alias("periodo_recepcion")))


def generar_afc_eventos(verdad: DataFrame) -> DataFrame:
    """Eventos del Seguro de Cesantia (AFC): inicio y termino de relacion laboral, conocidos en m+1."""
    return (verdad.filter(F.col("evento_afc").isNotNull())
            .select("afiliado_id", "periodo", "empleador_id", F.col("evento_afc").alias("tipo_evento"),
                    col_add_months(F.col("periodo"), 1).cast("int").alias("periodo_recepcion")))


def generar_todo(spark: SparkSession, cfg: Config) -> dict[str, DataFrame]:
    """Genera todas las tablas fuente de la simulacion."""
    from .calendario import macro_spark

    afiliados = generar_afiliados(spark, cfg)
    empleadores = generar_empleadores(spark, cfg)
    verdad = simular_verdad_mensual(spark, cfg, afiliados, empleadores)
    cotizaciones = generar_cotizaciones(cfg, verdad, empleadores)
    return {
        "afiliados": afiliados,
        "empleadores": empleadores,
        "verdad_mensual": verdad,
        "cotizaciones": cotizaciones,
        "licencias": generar_licencias(verdad),
        "afc_eventos": generar_afc_eventos(verdad),
        "macro": macro_spark(spark, cfg.periodo_inicio, cfg.periodo_snapshot),
    }
