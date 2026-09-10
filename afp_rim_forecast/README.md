# Nowcasting de Renta Imponible (RIM) para afiliados de una AFP — PySpark

Modelo que **proyecta la Renta Imponible Mensual de cada afiliado para los 2 meses que la AFP aún no conoce**
(la cotización llega ~2 meses después de la remuneración) y **reemplaza la proyección por el dato real** cuando
el empleador declara/paga. Todo el pipeline corre sobre PySpark (tablas de decenas de millones de filas) y se
simula aquí con datos sintéticos realistas.

```
python -m afp_rim_forecast.pipeline demo --n-afiliados 5000 --salida /tmp/afp_rim_forecast
```

---

## 1. El problema

| Concepto | Definición usada en el código |
|---|---|
| `periodo_snapshot` (T) | Mes en que corre el modelo ("hoy"). |
| Información conocida | Movimientos con `periodo_recepcion <= T-1` (todo lo recibido hasta fin del mes anterior). |
| `periodo_corte` | `T - meses_desfase` = **T-2**: último mes de remuneración completamente conocido. El pago normal del mes *m* llega en *m+1*, se procesa y queda disponible en *m+2*. |
| Meses abiertos | `corte+1` (h=1, es T-1) y `corte+2` (h=2, es T). Son **los 2 únicos meses que se proyectan**. |
| Ciclo mensual | En T+1 llegan los pagos del mes T-1 → la fila PREDICHA se reemplaza por REAL; el mes T se re-proyecta con h=1; aparece un nuevo mes T+1 con h=2. |

La RIM real se define a partir de los **movimientos** que recibe la AFP (planillas Previred, SII, subsidios):
por cada `(afiliado, periodo, entidad pagadora, empleador)` vale el último movimiento recibido (una
**rectificación** reemplaza a la declaración), se suma sobre pagadores y se aplica el **tope imponible** del
periodo. Una **DNP** (declaración y no pago) informa RIM igual; el pago posterior no la cambia.
Esa misma función (`snapshot.rim_conocida`) sirve para la "verdad final" (`hasta_recepcion=None`) y para la
"verdad conocida en un snapshot", que es la base de todas las features. **Nunca se mira la verdad latente.**

## 2. Datos: qué tablas tiene la AFP y cómo se simulan

| Tabla | Grano | Fuente real | Uso |
|---|---|---|---|
| `cotizaciones` (movimientos) | afiliado × periodo × pagador × movimiento | Previred / recaudación (DECLARACION, RECTIFICACION, PAGO_DNP; `estado_pago` PAGADA/DNP; `periodo_recepcion`) | Verdad as-of, lags, comportamiento de pago del empleador |
| `afiliados` | afiliado | Maestro de afiliados | Sexo, edad, región, tipo (DEPENDIENTE / INDEPENDIENTE / VOLUNTARIO / PENSIONADO_ACTIVO), nivel educacional, fecha afiliación |
| `empleadores` | empleador | Maestro de empleadores (RUT, CIIU, tamaño) | Rubro, tamaño, región |
| `afc_eventos` | afiliado × periodo × evento | Seguro de Cesantía (AFC): inicio/término de relación laboral (el empleador avisa en ~10 días) | **Señal adelantada** de cesantía / nuevo empleo |
| `licencias` | afiliado × periodo | COMPIN / ISAPRE (licencias médicas) | Subsidio: parte de la RIM llega por otro pagador y con más rezago |
| `macro` | periodo | Banco Central / INE / SP | UF, tope imponible (UF y CLP), IMM, IPC 12m, desempleo |
| `verdad_mensual` | afiliado × periodo | *(sólo simulación)* | Verdad latente para generar movimientos; **no se usa en el modelo** |

### Dinámicas que reproduce el generador (`synthetic.py`)

* **Empleo**: cadena de estados (empleado / cesante / licencia / retirado) con probabilidades de término y
  contratación por rubro y mes (temporeros en agricultura Nov–Mar, retail en Nov–Dic, educación con
  términos en Ene–Feb y contrataciones en Mar, turismo en verano), contratos plazo fijo con mayor rotación,
  antigüedad que reduce la rotación, edad de retiro 60/65.
* **Sueldo**: log-normal por productividad × rubro × empleador, piso legal en el **IMM** (con masa de
  afiliados exactamente en el mínimo), **reajustes** anuales/semestrales por IPC (sector público en
  diciembre), variación del IMM (mayo/julio/enero).
* **Componentes imponibles**: gratificación legal mensual (25 % con tope 4,75 IMM) o anual (abril),
  aguinaldos de Fiestas Patrias y Navidad, remuneración variable por rubro (comisiones en comercio, turnos
  en salud, bonos en minería/finanzas en marzo), horas extra, meses parciales al entrar/salir.
* **Tope imponible** (81,6 → 89,9 UF) aplicado por pagador y en la suma.
* **Múltiples empleadores** (5 %), **licencias médicas** (la ISAPRE/FONASA paga el subsidio con 2–4 meses
  de rezago), **independientes** (toda la renta del año llega vía SII en junio del año siguiente),
  **voluntarios** (pagos esporádicos), **pensionados que siguen cotizando**.
* **Comportamiento de pago**: por tamaño de empleador, pago anticipado (0–6 %), oportuno m+1 (84–97 %),
  tardío m+2, rezagos m+3..m+8, **DNP** (0–6 %) regularizada meses después, **rectificaciones** (1–2 %).
* **Afiliados nuevos** durante la ventana (cold start) y valores macro aproximados 2022–2026.

Todo se genera con expresiones Spark (`spark.range` + `rand`) y `mapInPandas` (simulación numpy por
partición): sin colecciones al driver ni cross joins, por lo que escala a millones de afiliados cambiando
`n_afiliados` y `particiones_simulacion`.

## 3. Variables del modelo (features "as-of")

Todas se calculan con `construir_features(snapshot)` usando **sólo** movimientos con
`periodo_recepcion <= snapshot-1`. En producción esta tabla se calcula una vez al mes y se persiste
particionada por `periodo_snapshot`; el set de entrenamiento es la unión de snapshots pasados con la RIM
real que llegó después (nunca se recalculan features "hacia atrás", lo que elimina el leakage por
construcción).

| Familia | Features | Por qué importa |
|---|---|---|
| Historia propia | `rim_l1..rim_l12` (l1 = mes de corte), `media_3/6/12`, `std_6`, `max_12`, `min_12`, `n_cotiza_3/6/12` (densidad), `meses_historia`, `meses_desde_ult_cotiza`, `meses_desde_cambio`, `ratio_l1_media12`, `tendencia_3_12` | La RIM es muy persistente; la densidad y el tiempo desde el último cambio capturan estabilidad laboral |
| Referencia | `rim_ref` (última RIM > 0), `rim_ref_sobre_imm`, `rim_ref_sobre_tope`, `rim_mismo_mes_ly` (valor en target-12), `ratio_ly_ref` | El target del regresor es `log(RIM_target / rim_ref)`; el mismo mes del año anterior captura aguinaldos, gratificación anual, bono de marzo |
| Relación laboral | `meses_con_empleador`, `n_empleadores_12`, `n_pagadores_l1`, `tiene_subsidio_l1`, `tiene_dnp_l1`, `tipo_contrato` (proxy por antigüedad) | Rotación y multiempleo |
| Señales adelantadas | `afc_termino_l1/l2`, `afc_inicio_l1/l2`, `meses_desde_afc_termino/inicio`, `dias_licencia_l1/l2`, `licencia_en_curso` | Un término AFC en el mes de corte implica RIM 0 el mes siguiente aunque el corte tenga RIM > 0 |
| Empleador | `rubro`, `tamano_empleador`, `emp_n_trabajadores`, `emp_crecimiento_12`, `emp_rim_mediana`, `emp_tasa_oportuno`, `emp_frac_declarado_corte`, `emp_frac_declarado_h1`, `sin_declaracion_con_emp_activo` | Si el empleador ya declaró el mes para el 90 % de sus trabajadores y no para este afiliado, es más probable un finiquito que un rezago; `emp_frac_declarado_h1` usa los pagos anticipados ya recibidos para el mes abierto |
| Afiliado | `tipo_afiliado`, `sexo`, `edad`, `region`, `nivel_educacional`, `meses_desde_afiliacion` | Segmentos con dinámicas distintas (independientes, voluntarios, pensionados, cold start) |
| Calendario y macro | `horizonte`, `mes_target`, `es_enero/marzo/abril/julio/sept/dic`, `imm_target`, `tope_target`, `ratio_imm_target_corte`, `ipc_12m_corte`, `desempleo_corte` | Estacionalidad legal (aguinaldos, gratificación, reajustes) y cambios del IMM ya legislados |

Filas ya conocidas para un mes abierto (empleadores que pagan anticipado) **no se predicen**: entran como
`REAL / PAGO_ANTICIPADO`.

## 4. Modelo (`model.py`)

**Hurdle en dos etapas** porque ~25–35 % de los afiliado-mes son 0 (cesantía, informalidad, independientes
sin renta):

1. `GBTClassifier` → `prob_cotiza = P(RIM_target > 0)`.
2. `GBTRegressor` (pérdida absoluta, robusta a outliers) sobre filas con RIM > 0 →
   `y_reg = log(RIM_target / rim_ref)`. Modelar el **ratio** y no el nivel hace que el modelo aprenda
   "cambios" (reajuste, aguinaldo, cambio de empleador) y generalice entre tramos de renta.

Salidas por afiliado y mes abierto:

* `rim_condicional = rim_ref · exp(ŷ)` topada al tope imponible del mes.
* `rim_esperada = prob_cotiza · rim_condicional` → para **agregados** (recaudación proyectada, flujo de fondos).
* `rim_proyectada = rim_condicional si prob_cotiza ≥ 0,5, si no 0` → para el **dato individual** que se
  escribe en la cuenta hasta que llegue el real.

Un horizonte es una feature (`horizonte` ∈ {1, 2}) y cada afiliado aporta dos filas por snapshot, lo que
comparte información entre horizontes sin duplicar modelos. Las categóricas pasan por `StringIndexer`
(`handleInvalid="keep"` para categorías nuevas en producción). En producción se sustituye GBT de MLlib por
`xgboost.spark.SparkXGBRegressor` / SynapseML LightGBM con la misma interfaz `fit/transform`.

## 5. Evaluación (`evaluate.py`, `pipeline.backtest`)

* **Orígenes móviles**: se entrena con `n_snapshots_entrenamiento` snapshots anteriores a
  `test - gap_entrenamiento` (gap de 3 meses para que los targets de entrenamiento estén realmente
  conocidos al entrenar) y se evalúa en los `n_snapshots_test` últimos.
* La verdad de entrenamiento es la **conocida al momento de entrenar** (`rim_conocida(hasta = test-1)`);
  la verdad de evaluación es la final.
* **Baselines** que hay que superar: persistencia (`rim_l1`), media de 3 meses, estacional
  (mismo mes del año anterior reajustado por la variación del IMM).
* **Métricas** por horizonte, tipo de afiliado, tramo de renta y snapshot: MAE, RMSE, **WAPE**
  (Σ|e| / Σ|y|, robusta a ceros; el MAPE sólo se reporta condicional a RIM > 0), sesgo agregado (error de la
  masa imponible proyectada), exactitud cero/no-cero, AUC y Brier del clasificador.

## 6. Reemplazo PREDICHA → REAL (`predict.py`, `reconcile.py`)

Tabla `rim_proyectada` (una fila por afiliado y periodo abierto):

| Columna | Descripción |
|---|---|
| `rim_valor` | Valor vigente (predicho o real) |
| `origen` | `PREDICHA` / `REAL` |
| `detalle_origen` | `MODELO`, `PAGO_ANTICIPADO`, `PAGO`, `REZAGO`, `RECTIFICACION`, `SIN_COTIZACION` |
| `prob_cotiza`, `rim_condicional`, `rim_esperada` | Salidas del modelo |
| `rim_predicha_previa` | Lo que decía el modelo antes del reemplazo (monitoreo del error) |
| `periodo_snapshot`, `periodo_reemplazo`, `version_modelo`, `fecha_calculo` | Trazabilidad |

Ciclo mensual en el snapshot T+1:

1. `reconciliar`: las PREDICHA cuyo periodo ya tiene cotización recibida pasan a REAL (`PAGO` si llegó en
   m+1, `REZAGO` si tardó más); una REAL cuyo monto cambió (rectificación, segundo empleador que pagó tarde)
   se actualiza; un periodo más allá de la ventana de rezago sin cotización se fija en 0
   (`SIN_COTIZACION`) y si después llega un rezago vuelve a actualizarse. Idempotente.
2. `actualizar_proyecciones`: MERGE con las nuevas proyecciones — REAL gana a PREDICHA y entre PREDICHAS gana
   el snapshot más nuevo (el h=2 del mes pasado pasa a h=1 con más información).
3. `reporte_reemplazos`: error del modelo sobre las filas recién reemplazadas (MAE, WAPE, sesgo, exactitud
   cero) → monitoreo continuo y disparador de reentrenamiento.

El equivalente en Delta Lake / Iceberg está documentado en `reconcile.py` (`MERGE INTO ... WHEN MATCHED AND
origen='PREDICHA' ... WHEN NOT MATCHED INSERT`).

## 7. Escala (tablas gigantes)

* **Particionado** de `cotizaciones` por `periodo_recepcion` (lo que llega cada mes es un append) y de
  `rim_proyectada` / `features` por `periodo_snapshot`; Z-ORDER/bucketing por `afiliado_id`.
* Lags y rolling con `Window.partitionBy(afiliado_id).orderBy(periodo)` sobre la **grilla densa**
  (afiliado × mes desde su afiliación) construida con `sequence + explode` por fila (sin cross join).
* Dimensiones (`empleadores`, `macro`) se difunden con `broadcast`; las agregaciones por empleador se hacen
  una vez por snapshot.
* Features **incrementales**: cada mes se calcula sólo el snapshot nuevo; el entrenamiento lee snapshots
  persistidos. Skew: los grandes empleadores no son problema porque las windows van por afiliado.
* Entrenamiento distribuido con MLlib (demo) o XGBoost-on-Spark / LightGBM; scoring batch de ~2 × N filas.
* Orquestación mensual (Airflow): `ingesta → reconciliar → features(T) → proyectar(T) → merge → reporte`;
  reentrenamiento trimestral o cuando el error de los reemplazos supera un umbral.

## 8. Cómo correr

```bash
pip install -r afp_rim_forecast/requirements.txt          # PySpark 4 necesita Java 17/21
python -m afp_rim_forecast.pipeline demo --n-afiliados 5000 --salida /tmp/afp_rim_forecast
python -m pytest afp_rim_forecast/tests -q
```

Salidas en `--salida`: `fuentes/` (tablas sintéticas en parquet), `features/periodo_snapshot=*`,
`modelo/`, `rim_proyectada/` (proyección del snapshot), `rim_proyectada_<T+1>/` (tras el ciclo mensual) y
`reportes/*.csv` con métricas, importancias y reemplazos.

Otros comandos: `generar`, `backtest`, `entrenar`, `proyectar` (usar `--reusar-fuentes` para no regenerar).

## 9. Estructura

```
afp_rim_forecast/
  config.py      parámetros (desfase, horizontes, ventana de entrenamiento, GBT)
  calendario.py  periodos yyyymm y tabla macro (UF, IMM, tope, IPC, desempleo)
  synthetic.py   generador de datos sintéticos (dims + mapInPandas + movimientos con rezago)
  snapshot.py    RIM conocida "as-of", grilla densa, filas ya conocidas de meses abiertos
  features.py    feature engineering sin leakage (una fila por afiliado × horizonte)
  model.py       modelo hurdle GBT (clasificador + regresor de log-ratio)
  evaluate.py    baselines y métricas
  predict.py     scoring → tabla rim_proyectada
  reconcile.py   reemplazo PREDICHA → REAL y MERGE mensual
  pipeline.py    CLI y demo end-to-end
  tests/         pytest con Spark local
```

## 10. Resultados del demo

*(se completan con la corrida de `demo`; ver `reportes/`)*

## 11. Supuestos y próximos pasos

* Los parámetros macro y las proporciones (rotación, DNP, rezagos) son supuestos razonables para Chile,
  no series oficiales; en producción se reemplazan por los datos reales de la AFP y los ajustes se validan
  con el backtest.
* Mejoras naturales: modelo específico para independientes (anual, vía Operación Renta), cuantiles
  (P10/P90) para intervalos, calibración del umbral por segmento según el costo de cada error,
  reentrenamiento automático por drift del reporte de reemplazos, y explicabilidad (SHAP) para auditoría.
