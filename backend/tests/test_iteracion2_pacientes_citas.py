"""
Pruebas Automatizadas de Validación — Paso 1 (HU-003 a HU-008).
"""

import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(".env")
from services import paciente_service, cita_service
from schemas.paciente_schema import PacienteCreate
from schemas.cita_schema import CitaCreate, CitaUpdate
from fastapi import HTTPException


def test_hu003_validacion_fecha_futura():
    """HU-003: La fecha de nacimiento no puede ser una fecha futura."""
    fecha_futura = date.today() + timedelta(days=10)
    try:
        PacienteCreate(
            nombre="Prueba",
            primer_apellido="Futura",
            fecha_nacimiento=fecha_futura
        )
        raise AssertionError("Se esperaba excepción por fecha futura")
    except ValueError as e:
        assert "no puede ser una fecha futura" in str(e).lower()
        print("[OK] HU-003: Fecha de nacimiento futura rechazada correctamente")


def test_hu004_historial_jsonb_paciente():
    """HU-004: Guardar automáticamente la versión anterior en historial JSONB."""
    # 1. Crear paciente de prueba
    datos_creacion = {
        "nombre": "TestHistorial",
        "primer_apellido": "Paciente",
        "telefono": "70000001",
        "email": "test.inicial@hospital.com",
        "ci": f"CI-{int(datetime.now().timestamp())}"
    }
    p = paciente_service.crear_paciente(datos_creacion)
    paciente_id = str(p["id"])

    # 2. Modificar teléfono y correo
    nuevo_tel = "79999999"
    nuevo_email = "test.actualizado@hospital.com"
    paciente_service.actualizar_paciente(paciente_id, {
        "telefono": nuevo_tel,
        "email": nuevo_email
    })

    # 3. Comprobar que en la base de datos se guardó el historial JSONB
    paciente_db = paciente_service.obtener_paciente(paciente_id)
    assert paciente_db["telefono"] == nuevo_tel
    assert paciente_db["email"] == nuevo_email
    assert isinstance(paciente_db["historial"], list)
    assert len(paciente_db["historial"]) >= 1

    ultimo_historial = paciente_db["historial"][-1]
    assert "fecha_modificacion" in ultimo_historial
    assert ultimo_historial["version_anterior"]["telefono"] == "70000001"
    assert ultimo_historial["version_anterior"]["email"] == "test.inicial@hospital.com"

    # Limpiar paciente de prueba
    paciente_service.desactivar_paciente(paciente_id)
    print("[OK] HU-004: Historial de paciente guardado automáticamente en JSONB")


def test_hu006_historial_jsonb_antecedentes():
    """HU-006: Conservar versión anterior de antecedentes médicos en historial JSONB."""
    # 1. Crear paciente temporal
    p = paciente_service.crear_paciente({
        "nombre": "TestAnt",
        "primer_apellido": "Ginecologia",
        "ci": f"CI-ANT-{int(datetime.now().timestamp())}"
    })
    paciente_id = str(p["id"])

    # 2. Crear antecedentes iniciales
    paciente_service.crear_antecedentes({
        "paciente_id": paciente_id,
        "edad_menarquia": 13,
        "tipo_ciclo": "regular",
        "duracion_ciclo_habitual": 28,
        "diabetes": False
    })

    # 3. Actualizar antecedentes
    paciente_service.actualizar_antecedentes(paciente_id, {
        "tipo_ciclo": "irregular",
        "duracion_ciclo_habitual": 35,
        "diabetes": True
    })

    # 4. Verificar historial JSONB
    ant_db = paciente_service.obtener_antecedentes(paciente_id)
    assert ant_db["tipo_ciclo"] == "irregular"
    assert ant_db["diabetes"] is True
    assert isinstance(ant_db["historial"], list)
    assert len(ant_db["historial"]) >= 1

    reg_hist = ant_db["historial"][-1]
    assert "fecha_modificacion" in reg_hist
    assert reg_hist["version_anterior"]["tipo_ciclo"] == "regular"
    assert reg_hist["version_anterior"]["duracion_ciclo_habitual"] == 28

    paciente_service.desactivar_paciente(paciente_id)
    print("[OK] HU-006: Historial de antecedentes médicos guardado en JSONB")


def test_hu007_validacion_proxima_cita_posterior():
    """HU-007: La fecha tentativa de próxima cita debe ser posterior a la fecha de atención."""
    fecha_atencion = datetime.now(timezone.utc) + timedelta(days=5)
    fecha_proxima_invalida = (fecha_atencion - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        cita_service.crear_cita(CitaCreate(
            fecha_atencion=fecha_atencion,
            fecha_proxima_cita=fecha_proxima_invalida,
            motivo="Control SOP"
        ))
        raise AssertionError("Se esperaba error por fecha próxima cita anterior a atención")
    except HTTPException as e:
        assert "posterior" in e.detail.lower()
        print("[OK] HU-007: Fecha tentativa de próxima cita validada como posterior")


def test_hu008_actualizacion_estados_cita():
    """HU-008: Estados disponibles (programada, atendida, cancelada, no asistió)."""
    # 0. Crear paciente de prueba
    p = paciente_service.crear_paciente({
        "nombre": "TestCita",
        "primer_apellido": "Estados",
        "ci": f"CI-CITA-{int(datetime.now().timestamp())}"
    })
    paciente_id = str(p["id"])

    # 1. Crear cita con estado inicial 'programada'
    cita = cita_service.crear_cita(CitaCreate(
        paciente_id=paciente_id,
        fecha_atencion=datetime.now(timezone.utc) + timedelta(days=2),
        motivo="Prueba estados",
        estado="programada"
    ))
    cita_id = cita["id"]
    assert cita["estado"] == "programada"

    # 2. Probar cambio a los estados válidos
    for nuevo_est in ["atendida", "cancelada", "no asistió"]:
        act = cita_service.actualizar_cita(cita_id, CitaUpdate(estado=nuevo_est))
        assert act["estado"] == nuevo_est

    # 3. Comprobar en base de datos
    cita_db = cita_service.obtener_cita_por_id(cita_id)
    assert cita_db["estado"] == "no asistió"

    cita_service.eliminar_cita(cita_id)
    paciente_service.desactivar_paciente(paciente_id)
    print("[OK] HU-008: Transición y guardado de los 4 estados de cita validado")


if __name__ == "__main__":
    print("\nEjecutando suite de pruebas para Paso 1 (HU-003 a HU-008)...")
    test_hu003_validacion_fecha_futura()
    test_hu004_historial_jsonb_paciente()
    test_hu006_historial_jsonb_antecedentes()
    test_hu007_validacion_proxima_cita_posterior()
    test_hu008_actualizacion_estados_cita()
    print("\n¡TODAS LAS PRUEBAS DEL PASO 1 (HU-003 a HU-008) PASARON EXITOSAMENTE!\n")
