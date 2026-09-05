"""Esquemas Pydantic de la Clasificación final de SOP.

STUB / PENDIENTE DE IMPLEMENTAR.

Aquí se definirán los modelos de datos que combinan los tres criterios de
Rotterdam para emitir la clasificación final del Síndrome de Ovario
Poliquístico (por ejemplo, cuántos criterios cumple la paciente y el
diagnóstico resultante).
"""

from pydantic import BaseModel


class ClasificacionResponse(BaseModel):
    """Resultado de la clasificación final de SOP (estructura provisional)."""

    # TODO: definir los campos reales una vez se integren los 3 criterios.
    # Ejemplo tentativo:
    # criterio1: bool
    # criterio2: bool
    # criterio3: bool
    # criterios_cumplidos: int
    # diagnostico: str
    pass
