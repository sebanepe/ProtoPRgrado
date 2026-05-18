Pruebas de Regresión

Actualmente el repositorio no cuenta con una suite de regresión dedicada. Se recomienda crear una suite estable y rápida a partir de los casos críticos ya cubiertos por pruebas unitarias e integración.

Casos mínimos recomendados para regresión.

Código	Caso crítico	Tipo
RG-01	La API responde correctamente en health check.	Integración
RG-02	El login de usuario válido retorna token o respuesta esperada.	Integración
RG-03	Se crea/importa un dataset válido sin errores.	Integración
RG-04	El preprocesamiento transforma correctamente un dataset mínimo.	Integración
RG-05	El sistema rechaza datos con estructura inválida.	Unit/Integración
RG-06	El scoring genera un score para cada transacción de prueba.	Integración
RG-07	Se genera alerta cuando el score supera el umbral.	Integración
RG-08	No se genera alerta cuando el score no supera el umbral.	Integración
RG-09	La configuración de modelo activo y umbral se mantiene correctamente.	Integración
RG-10	El frontend renderiza dashboard, alertas y configuración sin errores.	Frontend

Estrategia de suite de regresión.

Se recomienda dividir la regresión en dos niveles:

Nivel	Frecuencia	Contenido
Regresión rápida / PR-gate	En cada Pull Request	10 a 15 casos críticos, sin entrenamiento pesado.
Regresión extendida / Nightly	Diaria o semanal	Flujos completos, más datasets, más validaciones y pruebas frontend adicionales.

La suite de regresión rápida no debería superar los 5 minutos de ejecución. La suite extendida puede tolerar una duración mayor, siempre que no bloquee el ciclo normal de desarrollo.

Ubicación de la suite: `backend/app/tests/regression/`

Nota: `RG-10` (frontend) se ejecuta en CI mediante los pasos de Node/Vitest configurados en `.github/workflows/ci.yml`.
