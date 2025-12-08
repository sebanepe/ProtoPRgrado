# Instrumentación de la tecnología — Ejemplo de referencia

Este repositorio contiene un ejemplo didáctico y autocontenido para la sección
"Instrumentación de la Tecnología" de un proyecto de detección de fraude financiero.

Resumen:

- `src/instrumentacion.py`: script principal con funciones modulares: `cargar_datos`, `preprocesar`, `entrenar_modelo`, `evaluar_modelo`, `guardar_resultados`.
- `requirements.txt`: dependencias necesarias.

Características principales:

- Simula un dataset típico (monto, hora, canal, tipo_transacción, ubicación, etiqueta_fraude).
- Preprocesamiento mínimo y balanceo con `SMOTE`.
- Entrenamiento de un `DecisionTreeClassifier` y evaluación (Precision, Recall, F1, AUC).
- Guarda resultados de métricas y parámetros en una base de datos usando SQLAlchemy.

Requisitos:

- Python 3.10+
- Instalar dependencias:

  ```powershell
  python -m venv .venv; .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

Conexión a PostgreSQL (opcional pero recomendado):

- Defina la variable de entorno `DATABASE_URL` con el dialecto SQLAlchemy.
  Ejemplo (PowerShell):

  ```powershell
  $env:DATABASE_URL = "postgresql+psycopg2://usuario:password@localhost:5432/mi_base"
  ```

- Si `DATABASE_URL` no está definida o falla la conexión, el script hará un fallback
  a un archivo SQLite local `results_fallback.db` para que el ejemplo sea reproducible.

Ejecutar el ejemplo:

```powershell
python src/instrumentacion.py
```

Salida esperada:

- Mensajes en consola con progreso y métricas.
- Un archivo `feature_importances.png` en la carpeta desde la que se ejecuta.
- Un registro insertado en la tabla `experiment_results` de la base de datos (Postgres o SQLite fallback).

Notas didácticas:

- El script es deliberadamente simple y está diseñado como apéndice técnico: explica cada paso,
  usa técnicas reales (SMOTE, train/test split, métricas relevantes) y muestra cómo persistir resultados.
- Para producción habría que añadir: pipeline serializable, pruebas unitarias, validación más estricta,
  gestión de secretos en lugar de variables de entorno, y registro de modelos (model registry).
