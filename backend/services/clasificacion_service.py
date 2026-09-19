"""
Servicio de Clasificación Fenotípica de Rotterdam y Cálculos Automáticos Hormonales/Clínicos.

Implementa los Requerimientos Funcionales:
  - RF-11: Cálculo automático de HOMA-IR ((glucosa * insulina) / 405) y relación LH/FSH (lh / fsh)
  - RF-13: Cálculo automático de IMC (peso / talla^2)
  - RF-20: Clasificación del Fenotipo SOP según la Escala de Rotterdam (A, B, C, D o Sin SOP)
  - RF-21: Visualización de resultados por cada uno de los 3 criterios
"""

from typing import Dict, Any, Optional

def calcular_homa_ir(glucosa_mg_dl: float, insulina_uIU_ml: float) -> float:
    """Calcula el índice HOMA-IR para evaluar resistencia a la insulina."""
    if glucosa_mg_dl <= 0 or insulina_uIU_ml <= 0:
        return 0.0
    return round((glucosa_mg_dl * insulina_uIU_ml) / 405.0, 2)

def calcular_relacion_lh_fsh(lh: float, fsh: float) -> float:
    """Calcula la relación LH/FSH. Un valor > 2.0 sugiere sospecha endocrina de SOP."""
    if fsh <= 0:
        return 0.0
    return round(lh / fsh, 2)

def calcular_imc(peso_kg: float, talla_m: float) -> float:
    """Calcula el Índice de Masa Corporal (IMC)."""
    if talla_m <= 0:
        return 0.0
    return round(peso_kg / (talla_m ** 2), 2)

def clasificar_fenotipo_rotterdam(
    cumple_criterio1: bool,  # Disfunción ovulatoria (Oligo/Amenorrea)
    cumple_criterio2: bool,  # Hiperandrogenismo (Clínico/Bioquímico o XGBoost)
    cumple_criterio3: bool,  # Morfología ecográfica (EfficientNet-B0)
) -> Dict[str, Any]:
    """
    Clasifica el fenotipo de SOP de acuerdo a los consensos internacionales de Rotterdam:
    
    Criterios Evaluados:
      - Criterio 1: Olio/Amenorrea (Disfunción ovulatoria)
      - Criterio 2: Hiperandrogenismo (Clínico/Bioquímico)
      - Criterio 3: Morfología Ecográfica Poliquística (EfficientNet)

    Regla Diagnóstica: Requiere al menos 2 de los 3 criterios.

    Fenotipos:
      - Fenotipo A (Clásico Completo)    : Criterio 1 + Criterio 2 + Criterio 3
      - Fenotipo B (Clásico Anovulatorio): Criterio 1 + Criterio 2
      - Fenotipo C (Ovulatorio)          : Criterio 2 + Criterio 3
      - Fenotipo D (No Hiperandrogénico) : Criterio 1 + Criterio 3
      - Sin SOP                          : Menos de 2 criterios
    """
    total_criterios = sum([cumple_criterio1, cumple_criterio2, cumple_criterio3])
    tiene_sop = total_criterios >= 2

    fenotipo = "Sin SOP"
    descripcion = "No cumple los criterios mínimos de Rotterdam (se requieren al menos 2 de 3 criterios)."

    if cumple_criterio1 and cumple_criterio2 and cumple_criterio3:
        fenotipo = "Fenotipo A"
        descripcion = "SOP Clásico Completo: Presenta disfunción ovulatoria, hiperandrogenismo y morfología ecográfica de ovario poliquístico."
    elif cumple_criterio1 and cumple_criterio2 and not cumple_criterio3:
        fenotipo = "Fenotipo B"
        descripcion = "SOP Clásico Anovulatorio: Presenta disfunción ovulatoria e hiperandrogenismo (ecografía normal o no concluyente)."
    elif not cumple_criterio1 and cumple_criterio2 and cumple_criterio3:
        fenotipo = "Fenotipo C"
        descripcion = "SOP Ovulatorio: Presenta hiperandrogenismo y morfología ecográfica de ovario poliquístico (ciclos ovulatorios regulares)."
    elif cumple_criterio1 and not cumple_criterio2 and cumple_criterio3:
        fenotipo = "Fenotipo D"
        descripcion = "SOP No Hiperandrogénico: Presenta disfunción ovulatoria y morfología ecográfica poliquística sin hiperandrogenismo."

    return {
        "tiene_sop": tiene_sop,
        "fenotipo": fenotipo,
        "descripcion": descripcion,
        "total_criterios_cumplidos": total_criterios,
        "desglose_criterios": {
            "criterio_1_disfuncion_ovulatoria": {
                "cumple": cumple_criterio1,
                "nombre": "Criterio 1 — Disfunción Ovulatoria (Oligomenorrea / Amenorrea)"
            },
            "criterio_2_hiperandrogenismo": {
                "cumple": cumple_criterio2,
                "nombre": "Criterio 2 — Hiperandrogenismo Clínico/Bioquímico (XGBoost)"
            },
            "criterio_3_ecografico": {
                "cumple": cumple_criterio3,
                "nombre": "Criterio 3 — Morfología Ecográfica Poliquística (EfficientNet-B0)"
            }
        }
    }
