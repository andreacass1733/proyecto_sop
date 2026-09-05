"""Router del Criterio 2 - Disfunción ovulatoria (oligo/anovulación).

STUB / PENDIENTE DE IMPLEMENTAR.

Definirá las rutas HTTP del segundo criterio de Rotterdam y delegará la lógica
en `services/criterio2_service.py` (modelo XGBoost pendiente).
"""

from fastapi import APIRouter

router = APIRouter(prefix="/criterio2", tags=["Criterio 2 - Disfunción ovulatoria"])


@router.get("/")
def info():
    """Ruta provisional que indica que el criterio aún no está implementado."""
    # TODO: implementar las rutas reales del Criterio 2.
    return {"mensaje": "Criterio 2 pendiente de implementación"}
