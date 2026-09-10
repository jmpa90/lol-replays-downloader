"""Nowcasting de Renta Imponible Mensual (RIM) para afiliados de una AFP.

Paquete PySpark que simula datos, construye features "as-of" (sin leakage),
entrena un modelo hurdle (P(cotiza) x E[RIM | cotiza]), proyecta los dos
meses abiertos (desfase de 2 meses) y reemplaza la proyeccion por el dato
real cuando el empleador paga.
"""

__version__ = "0.1.0"
