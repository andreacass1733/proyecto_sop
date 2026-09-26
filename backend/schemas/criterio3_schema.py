"""
Esquemas Pydantic del Criterio 3 — Criterio ecográfico.

Los esquemas Pydantic definen la forma exacta de los datos que entran
y salen de la API. Supabase devuelve diccionarios, Pydantic los valida
y los convierte a objetos tipados para el frontend.

Clases definidas:
    EstudioEcograficoCreate  → Datos que llegan al crear un estudio (solo consulta_id)
    PrediccionResponse       → Respuesta completa tras analizar una imagen con IA
    ValidarRequest           → Datos que envía el médico al validar un resultado
    ValidacionResponse       → Confirmación de la validación guardada
    EstudioEcograficoOut     → Registro completo de un estudio (para listados e historial)
    EstadisticasResponse     → Métricas del modelo para el dashboard
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# ENTRADA: Crear un estudio (solo se necesita el id de la consulta)
# ══════════════════════════════════════════════════════════════════════════════

class EstudioEcograficoCreate(BaseModel):
    """
    Datos mínimos requeridos para iniciar un análisis ecográfico.
    La imagen se envía por separado como archivo (multipart/form-data).
    """
    # UUID de la consulta médica a la que pertenece este estudio
    consulta_id: UUID = Field(..., description="ID de la consulta médica asociada")


# ══════════════════════════════════════════════════════════════════════════════
# SALIDA: Respuesta inmediata tras predecir
# ══════════════════════════════════════════════════════════════════════════════

class PrediccionResponse(BaseModel):
    """
    Respuesta completa que devuelve la API después de analizar
    una ecografía con el modelo EfficientNet-B0.

    Incluye:
    - Probabilidades de cada clase (SOP vs Normal) en valores 0-1 y en porcentaje
    - Resultado de la clasificación ("Cumple criterio" / "No cumple criterio")
    - Estimación del número de folículos detectados por OpenCV
    - URL del mapa de calor generado para visualización del médico
    - URL de la imagen original subida a Supabase Storage
    """
    # Identificador único del registro creado en Supabase
    id: Optional[UUID] = None

    # ID de la consulta médica a la que pertenece este estudio
    consulta_id: Optional[UUID] = None

    # URL pública de la imagen en Supabase Storage
    imagen_url: str

    # Nombre del archivo guardado en Storage (con UUID para evitar colisiones)
    imagen_nombre: str

    # Probabilidades como valor decimal entre 0 y 1
    prob_sop:    float = Field(..., ge=0.0, le=1.0, description="Probabilidad de SOP (0 a 1)")
    prob_normal: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de Normal (0 a 1)")

    # Las mismas probabilidades expresadas como porcentaje (más legible para el médico)
    prob_sop_porcentaje:    float = Field(..., description="Probabilidad de SOP en porcentaje")
    prob_normal_porcentaje: float = Field(..., description="Probabilidad de Normal en porcentaje")

    # Clasificación final del modelo
    resultado: str = Field(..., description="'Cumple criterio' o 'No cumple criterio'")

    # Estimación automática de folículos (puede ser None si OpenCV no pudo detectar)
    num_foliculos: Optional[int] = Field(
        None,
        description="Número estimado de folículos detectados por OpenCV. El médico puede corregirlo."
    )

    # URL del mapa de calor (saliency map) generado para visualización
    # Puede ser None si la generación del mapa falló (no bloquea el análisis principal)
    mapa_calor_url: Optional[str] = Field(
        None,
        description="URL del mapa de calor en Storage. None si no se pudo generar."
    )

    # Versión del modelo que generó esta predicción (para trazabilidad)
    version_modelo: str

    # Fecha y hora en que se creó el registro en la base de datos
    created_at: datetime

    @field_validator("resultado")
    @classmethod
    def validar_resultado(cls, v: str) -> str:
        """Garantiza que el resultado solo pueda ser uno de los dos valores válidos."""
        valores_validos = {"Cumple criterio", "No cumple criterio"}
        if v not in valores_validos:
            raise ValueError(f"resultado debe ser uno de: {valores_validos}")
        return v


# ══════════════════════════════════════════════════════════════════════════════
# ENTRADA: El médico valida un estudio
# ══════════════════════════════════════════════════════════════════════════════

class ValidarRequest(BaseModel):
    """
    Datos que envía el médico cuando revisa, evalúa y confirma la ecografía.
    """
    etiqueta_real: str = Field(
        ...,
        description="Diagnóstico del médico: 'SOP' o 'Normal'"
    )
    observacion_medico: Optional[str] = Field(
        None,
        max_length=500,
        description="Notas clínicas del médico sobre esta imagen (opcional)"
    )
    paciente_id: Optional[str] = None
    consulta_id: Optional[str] = None
    imagen_url: Optional[str] = None
    imagen_nombre: Optional[str] = None
    prob_sop: Optional[float] = None
    prob_normal: Optional[float] = None
    resultado: Optional[str] = None
    num_foliculos: Optional[int] = None
    mapa_calor_url: Optional[str] = None
    lado_ovario: Optional[str] = "izquierdo"

    @field_validator("etiqueta_real")
    @classmethod
    def validar_etiqueta(cls, v: str) -> str:
        """Garantiza que la etiqueta sea exactamente 'SOP' o 'Normal'."""
        valores_validos = {"SOP", "Normal"}
        if v not in valores_validos:
            raise ValueError(f"etiqueta_real debe ser 'SOP' o 'Normal', no '{v}'")
        return v


# ══════════════════════════════════════════════════════════════════════════════
# SALIDA: Confirmación de la validación guardada
# ══════════════════════════════════════════════════════════════════════════════

class ValidacionResponse(BaseModel):
    """
    Respuesta que confirma que la validación del médico fue guardada correctamente.
    """
    # ID del estudio que fue validado
    id: UUID

    # Confirma que el estudio ahora está marcado como validado
    validado: bool

    # La etiqueta que asignó el médico
    etiqueta_real: str

    # Observaciones del médico si las ingresó
    observacion_medico: Optional[str]

    # Fecha y hora de la última actualización del registro
    updated_at: datetime


# ══════════════════════════════════════════════════════════════════════════════
# SALIDA: Registro completo de un estudio (para listados e historial)
# ══════════════════════════════════════════════════════════════════════════════

class EstudioEcograficoOut(BaseModel):
    """
    Representación completa de un estudio ecográfico para mostrar
    en listados, historial de consultas o detalle de un estudio específico.

    Todos los campos opcionales pueden ser None si el estudio está incompleto
    (por ejemplo, si el mapa de calor no se pudo generar).
    """
    id:              UUID
    consulta_id:     UUID

    # Imagen de la ecografía original subida por el médico
    imagen_url:      Optional[str]
    imagen_nombre:   Optional[str]

    # Probabilidades calculadas por el modelo EfficientNet-B0
    prob_sop:        Optional[float]
    prob_normal:     Optional[float]

    # Resultado de la clasificación del criterio ecográfico
    resultado:       Optional[str]

    # Conteo estimado de folículos por OpenCV (izquierdo y derecho por separado)
    num_foliculos_izq: Optional[int] = Field(None, description="Folículos estimados en ovario izquierdo")
    num_foliculos_der: Optional[int] = Field(None, description="Folículos estimados en ovario derecho")

    # URL del mapa de calor generado para visualización del médico
    mapa_calor_url:  Optional[str] = Field(None, description="URL del mapa de calor en Supabase Storage")

    # Estado de validación médica
    validado:        bool

    # Etiqueta real asignada por el médico (None hasta que valide)
    etiqueta_real:   Optional[str]

    # Notas clínicas del médico sobre esta imagen
    observacion_medico: Optional[str]

    # Versión del modelo que generó la predicción (para trazabilidad)
    version_modelo:  Optional[str]

    # Timestamps del registro en la base de datos
    created_at: datetime
    updated_at: datetime


# ══════════════════════════════════════════════════════════════════════════════
# SALIDA: Estadísticas del modelo para el dashboard
# ══════════════════════════════════════════════════════════════════════════════

class EstadisticasResponse(BaseModel):
    """
    Métricas globales del Criterio 3 para mostrar en el panel de control del médico.

    Permite al médico ver en un vistazo:
    - Cuántos estudios se han realizado en total
    - Cuántos fueron revisados vs cuántos están pendientes
    - Qué tan preciso es el modelo según las validaciones reales
    - Si ya es posible reentrenar el modelo para mejorarlo
    """
    # Número total de ecografías analizadas por el sistema
    total_estudios: int

    # Ecografías que el médico ya revisó y validó
    total_validados: int

    # Ecografías pendientes de revisión médica
    total_pendientes: int

    # Precisión calculada con validaciones reales del médico
    # Es None si todavía no hay ningún estudio validado
    precision_real: Optional[float] = Field(
        None,
        description="Precisión calculada sobre estudios validados. None si no hay validaciones aún."
    )

    # Precisión obtenida durante el entrenamiento original del modelo (valor fijo)
    precision_entrenamiento: float = 0.9531

    # Cantidad de validaciones acumuladas desde el último reentrenamiento
    imagenes_validadas_nuevas: int = Field(
        ...,
        description="Validaciones acumuladas. Cuando llegue al umbral, se puede reentrenar."
    )

    # Número mínimo de validaciones requeridas para poder reentrenar
    umbral_reentrenamiento: int = 50

    # Indica si ya se puede lanzar el reentrenamiento del modelo
    listo_para_reentrenar: bool = Field(
        ...,
        description="True si imagenes_validadas_nuevas >= umbral_reentrenamiento"
    )