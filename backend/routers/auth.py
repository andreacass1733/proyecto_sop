"""
Router de Autenticación y Control de Acceso — Supabase Auth (JWT) + Doble Factor (2FA).
Endpoints correspondientes a la Iteración 1:
  - POST /auth/login       → HU-001 Paso 1: Validación de credenciales + emisión de 2FA
  - POST /auth/verify-2fa  → HU-001 Paso 2: Verificación de código 2FA y emisión de JWT
  - POST /auth/resend-2fa  → Reenvío de código de verificación 2FA
  - POST /auth/logout      → HU-002: Cierre de sesión e invalidación de JWT
  - GET  /auth/me          → Verificación de sesión activa y datos del especialista
"""

from fastapi import APIRouter, Header, Request, HTTPException
from typing import Optional

from schemas.auth_schema import (
    LoginRequest,
    InitialLoginResponse,
    Verify2FARequest,
    Resend2FARequest,
    LoginResponse,
    LogoutResponse,
    UserSessionResponse,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
)
from services import auth_service

router = APIRouter(prefix="/auth", tags=["Autenticación"])


@router.post("/login", response_model=InitialLoginResponse)
def login_endpoint(body: LoginRequest, request: Request):
    """
    HU-001: Valida correo y contraseña contra Supabase Auth.
    Obligatoriamente despacha código de Doble Factor (2FA) al correo electrónico.
    """
    client_ip = request.client.host if request.client else None
    return auth_service.iniciar_sesion(
        email=body.email,
        password=body.password,
        client_ip=client_ip,
    )


@router.post("/verify-2fa", response_model=LoginResponse)
def verify_2fa_endpoint(body: Verify2FARequest, request: Request):
    """
    HU-001 Paso 2: Verifica el código 2FA emitido.
    Si es válido, emite el token JWT de Supabase Auth y los datos del médico especialista.
    """
    client_ip = request.client.host if request.client else None
    return auth_service.verificar_2fa(
        session_id=body.session_id,
        code=body.code,
        client_ip=client_ip
    )


@router.post("/resend-2fa")
def resend_2fa_endpoint(body: Resend2FARequest):
    """
    Genera un nuevo código 2FA en caso de no haberlo recibido.
    """
    return auth_service.reenviar_2fa(session_id=body.session_id)


@router.post("/logout", response_model=LogoutResponse)
def logout_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    HU-002: Cierre de sesión del sistema.
    Invalida el token JWT en Supabase Auth y audita el cierre de sesión en base de datos.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Token de autenticación requerido")

    token = authorization.replace("Bearer ", "").strip()
    client_ip = request.client.host if request.client else None
    return auth_service.cerrar_sesion(token=token, client_ip=client_ip)


@router.get("/me", response_model=UserSessionResponse)
def get_current_user_endpoint(authorization: Optional[str] = Header(None)):
    """
    Verifica si el token JWT actual es válido y devuelve los datos del médico especialista.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Token de autenticación requerido")

    token = authorization.replace("Bearer ", "").strip()
    user = auth_service.verificar_token(token)
    medico_data = auth_service._obtener_datos_medico(user.id)

    return {
        "user_id": user.id,
        "email": user.email,
        "medico": medico_data,
    }


# ── Recuperación y Cambio de Contraseña ─────────────────────────────────────

@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password_endpoint(body: ForgotPasswordRequest, request: Request):
    """
    Solicita la emisión de un código de recuperación de contraseña enviado al correo.
    """
    client_ip = request.client.host if request.client else None
    return auth_service.solicitar_recuperacion_contrasena(
        email=body.email,
        client_ip=client_ip
    )


@router.post("/reset-password", response_model=ResetPasswordResponse)
def reset_password_endpoint(body: ResetPasswordRequest, request: Request):
    """
    Restablece y cambia la contraseña del especialista en Supabase Auth.
    """
    client_ip = request.client.host if request.client else None
    return auth_service.restablecer_contrasena(
        email=body.email,
        new_password=body.new_password,
        code=body.code,
        session_id=body.session_id,
        client_ip=client_ip
    )
