"""Router de Clasificación final de SOP.

STUB / PENDIENTE DE IMPLEMENTAR.

Combinará los resultados de los tres criterios de Rotterdam para emitir la
clasificación/diagnóstico final del Síndrome de Ovario Poliquístico.
Los esquemas asociados se definen en `schemas/clasificacion_schema.py`.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/clasificacion", tags=["Clasificación final SOP"])


@router.get("/")
def info():
    """Ruta provisional que indica que la clasificación aún no está lista."""
    # TODO: implementar la lógica que combina los 3 criterios de Rotterdam.
    return {"mensaje": "Clasificación final pendiente de implementación"}
