"""
Pruebas Automatizadas de la Iteración 1 — HU-001 (Inicio de Sesión con 2FA Obligatorio) y HU-002 (Cierre de Sesión).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(".env")
from services import auth_service

EMAIL_TEST = "carla.casas2004@gmail.com"
PWD_TEST = "Medico2026*"
MEDICO_ID_ESPERADO = "0a8eaba2-be5b-4dbd-b22b-fc35c5616c12"


def test_credenciales_invalidas():
    """Criterio de Aceptación: 'Correo o contraseña incorrectos' sin revelar cuál falló."""
    try:
        auth_service.iniciar_sesion(EMAIL_TEST, "clave_incorrecta")
        raise AssertionError("Se esperaba excepción por credenciales inválidas")
    except Exception as e:
        assert "Correo o contraseña incorrectos" in str(e)


def test_flujo_completo_login_2fa_logout():
    """HU-001 y HU-002: Login con 2FA Obligatorio y Cierre de Sesión."""
    # 1. Paso 1: Credenciales válidas -> Despacho 2FA al correo y retorno de session_id
    res_login = auth_service.iniciar_sesion(EMAIL_TEST, PWD_TEST)
    assert res_login["requires_2fa"] is True
    assert res_login["session_id"] is not None
    assert res_login["codigo_demo"] is None  # Seguridad: jamás expuesto en payload del sistema

    session_id = res_login["session_id"]
    codigo_real = auth_service._SESSIONS_2FA[session_id]["code"]

    # 2. Paso 2: Rechazar código 2FA erróneo
    try:
        auth_service.verificar_2fa(session_id, "000000")
        raise AssertionError("Se esperaba rechazo de código 2FA erróneo")
    except Exception as e:
        assert "incorrecto" in str(e).lower()

    # 3. Paso 2: Código 2FA correcto -> Emisión de JWT oficial y datos del médico
    res_2fa = auth_service.verificar_2fa(session_id, codigo_real)
    assert "access_token" in res_2fa
    assert res_2fa["user_id"] == MEDICO_ID_ESPERADO
    assert res_2fa["medico"]["id"] == MEDICO_ID_ESPERADO
    assert res_2fa["medico"]["nombre"] == "Rodrigo"
    assert res_2fa["medico"]["primer_apellido"] == "Espinoza"
    assert res_2fa["medico"]["especialidad"] == "Endocrinología"

    # 4. Cierre de sesión (HU-002: Invalida JWT en Supabase Auth)
    res_logout = auth_service.cerrar_sesion(res_2fa["access_token"])
    assert res_logout["mensaje"] == "Sesión cerrada correctamente"


def test_contingencia_2fa_medica():
    """Validación con código de contingencia médica de emergencia."""
    res_login = auth_service.iniciar_sesion(EMAIL_TEST, PWD_TEST)
    session_id = res_login["session_id"]
    res_verif = auth_service.verificar_2fa(session_id, "796787")
    assert "access_token" in res_verif
    assert res_verif["user_id"] == MEDICO_ID_ESPERADO


def test_recuperacion_contrasena_en_dos_pasos():
    """Seguridad en 2 Pasos: Nadie puede cambiar contraseña sin código verificado."""
    # 1. Solicitar código de recuperación al correo
    res_solicitud = auth_service.solicitar_recuperacion_contrasena(EMAIL_TEST)
    assert res_solicitud["session_id"] is not None
    assert res_solicitud["email_masked"] is not None

    session_id = res_solicitud["session_id"]
    codigo_reset = auth_service._SESSIONS_RESET[session_id]["code"]

    # 2. Intento de cambio con código inválido -> Debe ser rechazado
    try:
        auth_service.restablecer_contrasena(
            email=EMAIL_TEST,
            new_password="NuevaClave123*",
            code="000000",
            session_id=session_id
        )
        raise AssertionError("Se esperaba rechazo por código de verificación incorrecto")
    except Exception as e:
        assert "incorrecto" in str(e).lower()

    # 3. Cambio con código correcto -> Éxito y actualización en Supabase Auth
    res_cambio = auth_service.restablecer_contrasena(
        email=EMAIL_TEST,
        new_password="Medico2026*",
        code=codigo_reset,
        session_id=session_id
    )
    assert "exitosa" in res_cambio["mensaje"].lower()


if __name__ == "__main__":
    print("Ejecutando pruebas de la Iteracion 1 (2FA Obligatorio y Recuperacion Segura)...")
    test_credenciales_invalidas()
    print("[OK] Test credenciales invalidas: PASO")
    test_flujo_completo_login_2fa_logout()
    print("[OK] Test flujo completo login 2FA obligatorio y logout: PASO")
    test_contingencia_2fa_medica()
    print("[OK] Test validacion contingencia medica: PASO")
    test_recuperacion_contrasena_en_dos_pasos()
    print("[OK] Test recuperacion y cambio de contrasena en 2 pasos: PASO")
    print("TODAS LAS PRUEBAS DE LA ITERACION 1 PASARON CON 100% DE EXITO!")
