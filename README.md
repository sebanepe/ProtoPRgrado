# sistema-deteccion-fraude

Proyecto prototipo académico para detección de fraude bancario utilizando machine learning.

Estructura básica creada para un prototipo con API en FastAPI, conexión a PostgreSQL mediante SQLAlchemy, validación con Pydantic y componentes modulares para modelos, esquemas, repositorios, servicios y ML.

Cómo ejecutar (desarrollo):

1. Crear y activar un entorno virtual (por ejemplo, `python -m venv .venv`).
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar la variable de entorno `DATABASE_URL` (opcional). Por defecto se usa:

```
postgresql://postgres:password@localhost:5432/fraud_db
```

4. Ejecutar la API:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Notas:
- Este repositorio contiene un prototipo académico y no está listo para producción.
- Las carpetas `data/`, `notebooks/` y `powerbi/` están reservadas para datos, experimentos y reportes.

Docker / Desarrollo (con Docker Compose):

- Requisitos: `docker` y `docker compose` instalados.
- Levantar servicios (Postgres, pgAdmin, backend):

```bash
docker compose up -d db pgadmin backend
```

- Inicializar la base de datos y sembrar datos (ejecutar dentro del contenedor `backend` para evitar problemas de puertos/credenciales en la máquina host):

```bash
docker compose exec backend python -m backend.app.init_db
```

- Alternativa (ejecución desde host):
	- Asegúrate de que el puerto mapeado en `docker-compose.yml` no esté siendo usado por una instancia local de Postgres. Si hay conflicto, cambia el mapeo (por ejemplo `"5433:5432"`) y actualiza `DATABASE_URL`.
	- Exportar la variable y ejecutar el script de inicialización desde tu venv (PowerShell ejemplo):

```powershell
$env:DATABASE_URL = "postgresql://protouser:protopass@127.0.0.1:5432/protodb"
.\.venv\Scripts\python.exe -m backend.app.init_db
```

- Ejecutar tests por interprete de venv (raiz de repo):

```powershell
.\.venv\Scripts\python -m pytest -q
```
- Ejecutar tests dentro del contenedor (rápido y fiable):

```bash
docker compose exec backend pytest -q
```

- Acceder a pgAdmin (por defecto mapeado en el host):
	- URL: http://localhost:8080
	- Usuario: el valor de `PGADMIN_DEFAULT_EMAIL` en `docker-compose.yml` (por defecto `sebanpb@gmail.com`)
	- Contraseña: el valor de `PGADMIN_DEFAULT_PASSWORD` en `docker-compose.yml` (por defecto `Mariokart8$`)

- Notas sobre autenticación y puertos:
	- Si tu máquina host ya ejecuta Postgres en el puerto `5432`, conecta el servicio `db` a otro puerto (por ejemplo `5433:5432`) para evitar que las conexiones desde el host sean atendidas por la instancia local.
	- Ejecutar `init_db` dentro del contenedor evita diferencias de `pg_hba.conf` y credenciales entre host y contenedor.
