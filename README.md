# fraud-detection-system

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
