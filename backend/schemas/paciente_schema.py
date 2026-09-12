"""
Esquemas Pydantic — Módulo de Pacientes y Antecedentes Médicos.

Define la validación de entrada y salida para:
  - Crear/editar pacientes
  - Listar pacientes con paginación
  - Crear/editar antecedentes médicos
"""

from datetime import date, datetime
from typing import Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# PACIENTE
# ══════════════════════════════════════════════════════════════════════════════

class PacienteCreate(BaseModel):
    """Datos para registrar un nuevo paciente."""
    nombre: str = Field(..., min_length=2, max_length=100)
    primer_apellido: str = Field(..., min_length=2, max_length=100)
    segundo_apellido: Optional[str] = Field(None, max_length=100)
    fecha_nacimiento: Optional[date] = None
    ci: Optional[str] = Field(None, max_length=20, description="Cédula de identidad única")
    telefono: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=150)
    medico_id: Optional[UUID] = Field(None, description="ID del médico tratante")


class PacienteUpdate(BaseModel):
    """Datos editables de un paciente (todos opcionales)."""
    nombre: Optional[str] = Field(None, min_length=2, max_length=100)
    primer_apellido: Optional[str] = Field(None, min_length=2, max_length=100)
    segundo_apellido: Optional[str] = Field(None, max_length=100)
    fecha_nacimiento: Optional[date] = None
    ci: Optional[str] = Field(None, max_length=20)
    telefono: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=150)
    medico_id: Optional[UUID] = None
    activo: Optional[bool] = None


class PacienteOut(BaseModel):
    """Respuesta completa de un paciente."""
    id: UUID
    nombre: str
    primer_apellido: str
    segundo_apellido: Optional[str] = None
    fecha_nacimiento: Optional[date] = None
    ci: Optional[str] = None
    telefono: Optional[str] = None
    email: Optional[str] = None
    medico_id: Optional[UUID] = None
    activo: bool
    created_at: datetime
    updated_at: datetime

    @property
    def nombre_completo(self) -> str:
        partes = [self.nombre, self.primer_apellido]
        if self.segundo_apellido:
            partes.append(self.segundo_apellido)
        return " ".join(partes)


class PacienteListItem(BaseModel):
    """Versión reducida para listados con email y médico asociado."""
    id: UUID
    nombre: str
    primer_apellido: str
    segundo_apellido: Optional[str] = None
    ci: Optional[str] = None
    telefono: Optional[str] = None
    email: Optional[str] = None
    fecha_nacimiento: Optional[date] = None
    activo: bool
    medico_id: Optional[UUID] = None
    created_at: datetime


# ══════════════════════════════════════════════════════════════════════════════
# ANTECEDENTES MÉDICOS
# ══════════════════════════════════════════════════════════════════════════════

class AntecedentesCreate(BaseModel):
    """Datos para crear los antecedentes médicos de un paciente."""
    paciente_id: UUID

    # Sexarca (Inicio de vida sexual)
    iniciada_vida_sexual: bool = Field(False, description="Indica si la paciente ha iniciado vida sexual")

    # Historial gineco-obstétrico
    edad_menarquia: Optional[int] = Field(None, ge=8, le=20, description="Edad de primera menstruación")
    tipo_ciclo: Optional[str] = Field(None, description="regular / irregular / amenorrea / oligomenorrea")
    duracion_ciclo_habitual: Optional[int] = Field(None, ge=1, le=90, description="Días del ciclo habitual o sangrado")

    # Obstétrico (solo aplica si iniciada_vida_sexual es True)
    gestas: int = Field(0, ge=0, description="Número de embarazos")
    partos: int = Field(0, ge=0)
    cesareas: int = Field(0, ge=0)
    abortos: int = Field(0, ge=0)
    hijos_vivos: int = Field(0, ge=0)

    # Anticonceptivos
    usa_anticonceptivos: bool = False
    tipo_anticonceptivo: Optional[str] = Field(None, max_length=100)

    # Antecedentes familiares
    familiar_con_sop: bool = False
    familiar_con_diabetes: bool = False

    # Antecedentes personales
    diabetes: bool = False
    hipotiroidismo: bool = False
    hiperprolactinemia: bool = False
    resistencia_insulina: bool = False

    # Notas y campo historial jsonb
    observaciones: Optional[str] = Field(None, max_length=1000)
    historial: Optional[Union[dict, list]] = None

    @field_validator("tipo_ciclo")
    @classmethod
    def validar_tipo_ciclo(cls, v):
        if v is not None and v not in {"regular", "irregular", "amenorrea", "oligomenorrea"}:
            raise ValueError("tipo_ciclo debe ser: regular, irregular, amenorrea u oligomenorrea")
        return v


class AntecedentesUpdate(BaseModel):
    """Actualización parcial de antecedentes médicos."""
    iniciada_vida_sexual: Optional[bool] = None
    edad_menarquia: Optional[int] = Field(None, ge=8, le=20)
    tipo_ciclo: Optional[str] = None
    duracion_ciclo_habitual: Optional[int] = Field(None, ge=1, le=90)
    gestas: Optional[int] = Field(None, ge=0)
    partos: Optional[int] = Field(None, ge=0)
    cesareas: Optional[int] = Field(None, ge=0)
    abortos: Optional[int] = Field(None, ge=0)
    hijos_vivos: Optional[int] = Field(None, ge=0)
    usa_anticonceptivos: Optional[bool] = None
    tipo_anticonceptivo: Optional[str] = None
    familiar_con_sop: Optional[bool] = None
    familiar_con_diabetes: Optional[bool] = None
    diabetes: Optional[bool] = None
    hipotiroidismo: Optional[bool] = None
    hiperprolactinemia: Optional[bool] = None
    resistencia_insulina: Optional[bool] = None
    observaciones: Optional[str] = None
    historial: Optional[Union[dict, list]] = None


class AntecedentesOut(BaseModel):
    """Respuesta completa de antecedentes médicos."""
    id: UUID
    paciente_id: UUID
    iniciada_vida_sexual: bool = False
    edad_menarquia: Optional[int] = None
    tipo_ciclo: Optional[str] = None
    duracion_ciclo_habitual: Optional[int] = None
    gestas: int = 0
    partos: int = 0
    cesareas: int = 0
    abortos: int = 0
    hijos_vivos: int = 0
    usa_anticonceptivos: bool = False
    tipo_anticonceptivo: Optional[str] = None
    familiar_con_sop: bool = False
    familiar_con_diabetes: bool = False
    diabetes: bool = False
    hipotiroidismo: bool = False
    hiperprolactinemia: bool = False
    resistencia_insulina: bool = False
    observaciones: Optional[str] = None
    historial: Optional[Union[dict, list]] = None
    created_at: datetime
    updated_at: datetime
