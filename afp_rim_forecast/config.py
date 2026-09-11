"""Configuracion central del pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Simulacion -------------------------------------------------------
    n_afiliados: int = 5_000
    afiliados_por_empleador: int = 40          # define n_empleadores
    periodo_inicio: int = 202201               # primer mes simulado (yyyymm)
    periodo_snapshot: int = 202609             # "hoy": mes en que corre el modelo
    seed: int = 42
    particiones_simulacion: int = 8

    # --- Desfase de informacion ------------------------------------------
    # En el snapshot T se conocen las cotizaciones RECIBIDAS hasta T-1
    # (fin del mes anterior). Como el pago normal del mes m llega en m+1,
    # el ultimo mes de remuneracion completo es T-2 => 2 meses abiertos.
    meses_desfase: int = 2
    horizontes: tuple[int, ...] = (1, 2)

    # --- Entrenamiento -----------------------------------------------------
    n_snapshots_entrenamiento: int = 12        # origenes moviles usados para entrenar
    gap_entrenamiento: int = 5                 # snapshot_train <= snapshot_test - gap (= desfase + maduracion)
    meses_maduracion: int = 3                  # meses extra de espera para pagos tardios y subsidios (lag <= 4)
    n_snapshots_test: int = 2                  # origenes reservados para backtest
    gbt_max_iter: int = 40
    gbt_max_depth: int = 5
    gbt_step_size: float = 0.1
    gbt_subsampling: float = 0.8
    version_modelo: str = "rim-hurdle-gbt-v1"

    # --- Salidas -----------------------------------------------------------
    ruta_salida: str = "/tmp/afp_rim_forecast"
    shuffle_partitions: int = 16

    # columnas categoricas que se indexan antes del modelo
    categoricas: tuple[str, ...] = field(
        default=("tipo_afiliado", "sexo", "rubro", "tamano_empleador", "tipo_contrato")
    )

    @property
    def n_empleadores(self) -> int:
        return max(20, self.n_afiliados // self.afiliados_por_empleador)

    @property
    def periodo_corte(self) -> int:
        """Ultimo mes de remuneracion completamente conocido en el snapshot."""
        from .calendario import add_months

        return add_months(self.periodo_snapshot, -self.meses_desfase)
