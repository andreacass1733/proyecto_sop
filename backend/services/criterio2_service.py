"""Servicio del Criterio 2 - Disfunción ovulatoria (oligo/anovulación).

STUB / PENDIENTE DE IMPLEMENTAR.

Este servicio contendrá la lógica del segundo criterio de Rotterdam. La idea es
usar un modelo de tipo XGBoost sobre variables clínicas (regularidad del ciclo,
etc.) para estimar la disfunción ovulatoria.
"""


def run_prediction(datos: dict):
    """Ejecuta la predicción del Criterio 2 a partir de datos clínicos.

    Args:
        datos (dict): variables clínicas de la paciente.

    Returns:
        dict: resultado de la predicción (por definir).
    """
    # TODO: cargar el modelo XGBoost del Criterio 2.
    # TODO: preprocesar `datos` al formato esperado por el modelo.
    # TODO: ejecutar la predicción y devolver probabilidad + resultado.
    raise NotImplementedError("El Criterio 2 aún no está implementado.")
