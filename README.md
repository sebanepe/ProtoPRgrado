# Sistema de Detección y Monitoreo de Fraude Bancario

Prototipo académico para detección de fraude bancario con machine learning. Incluye pipeline de datos (Fase A), alertas (Fase B), modelos supervisados/no supervisados (Fase C), monitoreo en tiempo real (Fase D) y administración de usuarios.

---

## Requisitos previos

| Herramienta | Versión mínima | Para qué |
|---|---|---|
| Docker | 24+ | Base de datos PostgreSQL y backend |
| Docker Compose | v2 (`docker compose`) | Orquestación de servicios |
| Python | 3.11+ | Entorno virtual y tests desde el host |
| Node.js | 18+ | Frontend React |

> El backend corre dentro de Docker. Python en el host solo es necesario para ejecutar tests localmente o scripts de migración desde fuera del contenedor.

---

## Instalación desde cero

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd ProtoPRgrado
```

### 2. Verificar el archivo `.env`

El repositorio incluye un `.env` con valores por defecto para desarrollo local:

```
DATABASE_URL=postgresql://protouser:protopass@127.0.0.1:5432/protodb
APP_ENV=development
SECRET_KEY=change_me
```

Este archivo es leído por el backend cuando se ejecuta desde el host. El contenedor Docker usa sus propias variables definidas en `docker-compose.yml` y no depende de este `.env`.

### 3. Crear entorno virtual Python (para tests y scripts en el host)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Instalar dependencias del frontend

```powershell
cd frontend
npm install
cd ..
```

---

## Levantar los servicios

### Paso 1: Iniciar la base de datos

```bash
docker compose up -d db
```

Espera unos segundos para que PostgreSQL esté listo. Puedes verificar con:

```bash
docker compose logs db --tail 10
```

Cuando veas `database system is ready to accept connections`, continúa.

### Paso 2: Iniciar el backend

```bash
docker compose up -d backend
```

El backend sirve la API en `http://localhost:8000`.

> El contenedor monta `./backend` como volumen (`./backend:/app/backend`), por lo que los cambios en el código Python se aplican sin reconstruir la imagen. Solo necesitas reiniciar el contenedor: `docker compose restart backend`.

### Paso 3: Inicializar la base de datos

Este comando crea todas las tablas, siembra roles, permisos y crea el usuario administrador por defecto:

```bash
docker compose exec backend python -m backend.app.init_db
```

Deberías ver una salida como:

```
Roles: ['ADMIN', 'DATA_SCIENTIST', 'FRAUD_ANALYST']
Permissions: ['dashboard.view', 'dataset.view', ...] ... total 29
Created default admin: sebanpb@gmail.com
```

> **Solo se necesita ejecutar `init_db` una vez** (o cada vez que quieras restablecer roles/permisos sin borrar datos). Es idempotente: no duplica registros si ya existen.

### Paso 4: Iniciar el frontend

```powershell
cd frontend
npm run dev
```

El frontend estará disponible en `http://localhost:5173`.

---

## Credenciales de acceso por defecto

| Campo | Valor |
|---|---|
| Email | `sebanpb@gmail.com` |
| Contraseña | `mariokart8$` |

Estas credenciales se configuran en `backend/app/init_db.py` mediante las variables de entorno `DEFAULT_ADMIN_EMAIL` y `DEFAULT_ADMIN_PASSWORD`. Puedes sobreescribirlas antes de ejecutar `init_db`.

---

## Migraciones de base de datos

`init_db.py` incluye todas las migraciones de manera idempotente (`ADD COLUMN IF NOT EXISTS`). Solo necesitas ejecutar el paso 3 de arriba y el esquema queda completo:

- **Instalación fresca**: `create_all` crea todas las tablas con el esquema actual, las funciones de migración son no-ops.
- **Base de datos existente con datos**: `create_all` no toca las tablas existentes, las funciones de migración agregan únicamente las columnas que faltan sin borrar datos.

En ambos casos: **un solo `docker compose exec backend python -m backend.app.init_db` es suficiente**.

Los archivos en `backend/migrations/` se mantienen como referencia histórica. No necesitas ejecutarlos manualmente.

---

## Ejecutar tests

### Backend (dentro del contenedor — recomendado)

```bash
docker compose exec backend pytest -q
```

Para correr solo tests de integración o unitarios:

```bash
docker compose exec backend pytest backend/app/tests/unit -q
docker compose exec backend pytest backend/app/tests/integration -q
```

### Backend (desde el host con venv activo)

```powershell
.\.venv\Scripts\python -m pytest -q
```

> Los tests usan SQLite en memoria por defecto. Para apuntar al PostgreSQL real del contenedor, establece `TEST_USE_REAL_DB=1` antes de correr pytest. Los tests **nunca escriben** en `data/processed` ni `data/uploads` del proyecto — las rutas se redirigen a directorios temporales automáticamente.

### Frontend

```powershell
cd frontend
npm test -- --run
```

Para build de producción:

```powershell
npm run build
```

---

## Acceso a pgAdmin

Interfaz web de administración de PostgreSQL, útil para inspeccionar datos:

- URL: `http://localhost:8080`
- Usuario: `sebanpb@gmail.com`
- Contraseña: `Mariokart8$`

Para agregar el servidor dentro de pgAdmin:
- Host: `db`
- Puerto: `5432`
- Base de datos: `protodb`
- Usuario: `protouser`
- Contraseña: `protopass`

---

## Referencia rápida de comandos

```bash
# Levantar todo
docker compose up -d db backend

# Ver logs del backend
docker compose logs backend -f

# Reiniciar backend tras cambios en código Python
docker compose restart backend

# Reconstruir imagen tras cambios en Dockerfile o requirements.txt
docker compose build backend && docker compose up -d backend

# Inicializar / restaurar roles, permisos y admin
docker compose exec backend python -m backend.app.init_db

# Ejecutar todos los tests en el contenedor
docker compose exec backend pytest -q

# Detener todos los servicios
docker compose down

# Detener y borrar volúmenes (borra todos los datos de PostgreSQL)
docker compose down -v
```

---

## Estructura del proyecto

```
ProtoPRgrado/
├── backend/
│   ├── app/
│   │   ├── main.py          # Punto de entrada FastAPI
│   │   ├── init_db.py       # Script de inicialización (tablas, roles, admin)
│   │   ├── models/          # Modelos SQLAlchemy
│   │   ├── routes/          # Endpoints por fase (dataset, preprocessing, etc.)
│   │   ├── services/        # Lógica de negocio
│   │   ├── repositories/    # Acceso a datos
│   │   ├── schemas/         # Esquemas Pydantic
│   │   ├── ml/              # Módulos de machine learning
│   │   └── tests/           # Tests unitarios, integración y regresión
│   ├── migrations/          # Scripts SQL idempotentes (ALTER TABLE IF NOT EXISTS)
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/           # Pantallas (Dashboard, Usuarios, etc.)
│   │   ├── components/      # Componentes reutilizables
│   │   └── services/api.js  # Cliente HTTP hacia el backend
│   └── package.json
├── data/
│   ├── uploads/             # CSVs subidos por el usuario (en Docker: /app/data/uploads)
│   └── processed/           # Resultados de preprocesamiento (en Docker: /app/data/processed)
├── docker-compose.yml
└── .env                     # Variables de entorno locales (no se usa dentro de Docker)
```

---

## Notas adicionales

- **Cambios en código Python**: solo requieren `docker compose restart backend` (el volumen monta el código en vivo).
- **Cambios en `requirements.txt` o `Dockerfile`**: requieren `docker compose build backend`.
- **Puerto 5432 ocupado**: si tu máquina ya tiene PostgreSQL local, cambia el mapeo en `docker-compose.yml` (por ejemplo `"5433:5432"`) y actualiza `DATABASE_URL` en `.env`.
- **Los tests son seguros**: nunca tocan la base de datos de producción ni el sistema de archivos del proyecto. Usan SQLite en memoria y directorios temporales.
- **Autenticación**: usa `Authorization: Bearer <token>` (JWT) o el header legacy `X-User-Email` para requests autenticados.
