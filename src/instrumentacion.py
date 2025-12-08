"""
Instrumentación de la tecnología - Ejemplo didáctico

Este script crea un flujo reducido y realista para instrumentar
un proyecto de detección de fraude financiero usando Python.

Funciones principales (cumpliendo la solicitud):
- cargar_datos()
- preprocesar()
- entrenar_modelo()
- evaluar_modelo()
- guardar_resultados()

El script intenta conectarse a PostgreSQL vía la variable de entorno
`DATABASE_URL` (ej: postgresql+psycopg2://user:pass@host:5432/dbname).
Si la conexión falla, usa un fallback a SQLite local para demostración.

Está documentado en español y contiene manejo básico de errores.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from imblearn.over_sampling import SMOTE

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker


# ---------------------------
# Configuración básica
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("instrumentacion")

Base = declarative_base()


class ExperimentResult(Base):
    """Modelo simple de tabla para almacenar resultados de experimentos.

    - `params` y `metrics` se guardan como JSON en texto para simplicidad.
    - Se crea la tabla si no existe.
    """

    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(100))
    params = Column(Text)
    metrics = Column(Text)
    notes = Column(Text)


def cargar_datos(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Simula un dataset pequeño con columnas típicas de ATC.

    Columnas:
    - `monto`: float, monto de la transacción
    - `hora`: int [0-23], hora del día
    - `canal`: categorical, canal de la transacción
    - `tipo_transaccion`: categorical
    - `ubicacion`: categorical (ciudad/region)
    - `etiqueta_fraude`: 0/1 con baja prevalencia

    El dataset es estocástico pero reproducible usando `random_state`.
    """
    rng = np.random.default_rng(random_state)

    # Monto con distribución asimétrica (más transacciones de bajo monto)
    monto = np.round(rng.exponential(scale=80, size=n_samples) + 1.0, 2)

    # Hora del día
    hora = rng.integers(0, 24, size=n_samples)

    # Categorías simuladas
    canales = ["web", "mobile", "branch", "call_center"]
    tipos = ["pago", "retiro", "transferencia", "consulta"]
    ubicaciones = ["La Paz", "El Alto", "Cochabamba", "Santa Cruz"]

    canal = rng.choice(canales, size=n_samples, p=[0.5, 0.3, 0.12, 0.08])
    tipo_transaccion = rng.choice(tipos, size=n_samples, p=[0.45, 0.2, 0.3, 0.05])
    ubicacion = rng.choice(ubicaciones, size=n_samples, p=[0.4, 0.25, 0.2, 0.15])

    # Generación de etiqueta de fraude con baja prevalencia y alguna dependencia en monto
    base_prob = 0.02  # prevalencia base 2%
    # Aumentar probabilidad si monto es grande o canal es remoto
    score = base_prob + (monto > 300).astype(float) * 0.05
    score += np.isin(canal, ["web", "mobile"]).astype(float) * 0.01

    etiqueta_fraude = rng.random(n_samples) < score

    df = pd.DataFrame(
        {
            "monto": monto,
            "hora": hora,
            "canal": canal,
            "tipo_transaccion": tipo_transaccion,
            "ubicacion": ubicacion,
            "etiqueta_fraude": etiqueta_fraude.astype(int),
        }
    )

    logger.info("Datos simulados: %d filas generadas", len(df))
    return df


def preprocesar(
    df: pd.DataFrame, smote: bool = True, test_size: float = 0.3, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Preprocesamiento mínimo.

    Pasos:
    - Limpieza simple (dropna)
    - Codificación de variables categóricas (One-Hot)
    - División train/test
    - Balanceo con SMOTE (si `smote=True`) aplicado solo al conjunto de entrenamiento

    Devuelve: X_train, X_test, y_train, y_test, info (metadatos útiles)
    """
    df_clean = df.copy()

    # Limpieza mínima
    initial_rows = len(df_clean)
    df_clean.dropna(inplace=True)
    if len(df_clean) != initial_rows:
        logger.warning("Se eliminaron %d filas con NA", initial_rows - len(df_clean))

    # Separar features y target
    target_col = "etiqueta_fraude"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # Codificación: One-Hot para variables categóricas
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("Columnas categóricas: %s", categorical_cols)
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "Split: X_train=%d, X_test=%d, positivos_train=%d, positivos_test=%d",
        len(X_train),
        len(X_test),
        int(y_train.sum()),
        int(y_test.sum()),
    )

    info = {
        "feature_names": X_encoded.columns.tolist(),
        "categorical_cols": categorical_cols,
        "original_shape": df.shape,
    }

    # Aplicar SMOTE solo si hay al menos una clase minoritaria adecuada
    if smote:
        try:
            sm = SMOTE(random_state=random_state)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            logger.info(
                "SMOTE aplicado: antes=%s, después=%s",
                y_train.value_counts().to_dict(),
                pd.Series(y_train_res).value_counts().to_dict(),
            )
            X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
            y_train = pd.Series(y_train_res)
        except Exception as e:
            logger.error("Error aplicando SMOTE: %s - continuando sin SMOTE", e)

    return X_train, X_test, y_train, y_test, info


def entrenar_modelo(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42, max_depth: int = 5
) -> DecisionTreeClassifier:
    """Entrena un Decision Tree simple y devuelve el modelo ajustado.

    Parámetros del modelo son retornados externamente si se desea guardarlos.
    """
    try:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        logger.info("Modelo entrenado: DecisionTree max_depth=%s", max_depth)
        return model
    except Exception as e:
        logger.exception("Error al entrenar el modelo: %s", e)
        raise


def evaluar_modelo(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
    plot_path: str = "feature_importances.png",
) -> Tuple[Dict[str, float], str]:
    """Evalúa el modelo y genera métricas clave y una visualización sencilla.

    Métricas calculadas: Recall, Precision, F1-score, AUC.
    Además guarda un gráfico de importancia de variables en `plot_path`.
    """
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
    auc = None
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            auc = metrics.roc_auc_score(y_test, y_proba)
        except Exception:
            auc = None

    metricas = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(auc) if auc is not None else None,
    }

    logger.info("Métricas: %s", metricas)

    # Gráfico de importancias (si el modelo lo soporta)
    try:
        importancias = model.feature_importances_
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(importancias))]

        imp_series = pd.Series(importancias, index=feature_names).sort_values(ascending=True)
        plt.figure(figsize=(8, max(4, len(imp_series) * 0.15)))
        imp_series.plot(kind="barh")
        plt.title("Importancia de variables - Decision Tree")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info("Grafico de importancias guardado en %s", plot_path)
    except Exception as e:
        logger.warning("No se pudo generar gráfico de importancias: %s", e)
        plot_path = ""

    return metricas, plot_path


def guardar_resultados(
    metrics: Dict[str, Any],
    params: Dict[str, Any],
    model_name: str = "DecisionTree",
    notes: str = "",
    db_url: str | None = None,
    table_name: str = "experiment_results",
) -> None:
    """Guarda métricas y parámetros en una base de datos usando SQLAlchemy.

    Intenta conectar a `db_url`. Si es None, intenta leer `DATABASE_URL`
    desde variables de entorno. Si la conexión falla, hace un fallback a SQLite
    local `results_fallback.db` para que el ejemplo sea reproducible.
    """
    if db_url is None:
        db_url = os.getenv("DATABASE_URL")

    if db_url is None:
        logger.warning("No se encontró DATABASE_URL: usando fallback SQLite local.")
        db_url = "sqlite:///results_fallback.db"

    engine = None
    Session = None
    try:
        engine = create_engine(db_url, future=True)
        Session = sessionmaker(bind=engine)
        # Crear tablas si no existen
        Base.metadata.create_all(engine)

        # Insertar resultado
        session = Session()
        row = ExperimentResult(
            model_name=model_name,
            params=json.dumps(params, default=str),
            metrics=json.dumps(metrics, default=str),
            notes=notes,
        )
        session.add(row)
        session.commit()
        session.close()
        logger.info("Resultados guardados en la base de datos: %s", db_url)
    except Exception as e:
        logger.exception("Error guardando resultados en la DB (%s): %s", db_url, e)
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass
        # Intentar fallback a SQLite si no se estaba usando ya
        if not db_url.startswith("sqlite"):
            fallback = "sqlite:///results_fallback.db"
            logger.info("Intentando fallback a %s", fallback)
            try:
                engine = create_engine(fallback, future=True)
                Base.metadata.create_all(engine)
                Session = sessionmaker(bind=engine)
                session = Session()
                row = ExperimentResult(
                    model_name=model_name,
                    params=json.dumps(params, default=str),
                    metrics=json.dumps(metrics, default=str),
                    notes=notes + " (fallback)",
                )
                session.add(row)
                session.commit()
                session.close()
                logger.info("Resultados guardados en fallback SQLite: %s", fallback)
            except Exception:
                logger.exception("No se pudo guardar ni en la DB principal ni en fallback.")


def main() -> None:
    """Orquestador: ejecuta el flujo completo de instrumentación.

    - Genera datos simulados
    - Preprocesa y balancea
    - Entrena Decision Tree
    - Evalúa y grafica
    - Guarda resultados en DB (Postgres preferido)
    """
    try:
        df = cargar_datos(n_samples=2000, random_state=123)

        X_train, X_test, y_train, y_test, info = preprocesar(df, smote=True)

        params = {"model": "DecisionTree", "max_depth": 5, "random_state": 42}
        model = entrenar_modelo(X_train, y_train, random_state=42, max_depth=params["max_depth"])

        metrics_res, plot_path = evaluar_modelo(
            model, X_test, y_test, feature_names=info["feature_names"], plot_path="feature_importances.png"
        )

        # Guardar resultados en DB; se leerá DATABASE_URL si existe
        notes = "Ejemplo didáctico: flujo reducido para instrumentación"
        guardar_resultados(metrics_res, params, model_name="DecisionTree", notes=notes)

        logger.info("Flujo completado. Métricas: %s", metrics_res)
        if plot_path:
            logger.info("Ver gráfico en: %s", plot_path)

    except Exception as e:
        logger.exception("Error en el flujo principal: %s", e)


if __name__ == "__main__":
    main()
