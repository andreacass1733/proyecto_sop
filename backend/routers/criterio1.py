"""Router del Criterio 1 - Hiperandrogenismo (clínico/bioquímico).

STUB / PENDIENTE DE IMPLEMENTAR.

Definirá las rutas HTTP del primer criterio de Rotterdam y delegará la lógica
en `services/criterio1_service.py` (modelo XGBoost pendiente).
"""

from fastapi import APIRouter

router = APIRouter(prefix="/criterio1", tags=["Criterio 1 - Hiperandrogenismo"])


@router.get("/")
def info():
    """Ruta provisional que indica que el criterio aún no está implementado."""
    # TODO: implementar las rutas reales del Criterio 1.
    return {"mensaje": "Criterio 1 pendiente de implementación"}
