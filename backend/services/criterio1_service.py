"""Servicio del Criterio 1 - Hiperandrogenismo (clínico/bioquímico).

STUB / PENDIENTE DE IMPLEMENTAR.

Este servicio contendrá la lógica del primer criterio de Rotterdam. La idea es
usar un modelo de tipo XGBoost sobre variables clínicas y bioquímicas para
estimar la presencia de hiperandrogenismo.
"""


def run_prediction(datos: dict):
    """Ejecuta la predicción del Criterio 1 a partir de datos clínicos.

    Args:
        datos (dict): variables clínicas/bioquímicas de la paciente.

    Returns:
        dict: resultado de la predicción (por definir).
    """
    # TODO: cargar el modelo XGBoost del Criterio 1.
    # TODO: preprocesar `datos` al formato esperado por el modelo.
    # TODO: ejecutar la predicción y devolver probabilidad + resultado.
    raise NotImplementedError("El Criterio 1 aún no está implementado.")
