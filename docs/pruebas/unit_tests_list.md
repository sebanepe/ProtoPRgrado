# Listado de pruebas unitarias

Este archivo agrupa las pruebas unitarias del proyecto (backend + frontend) y describe brevemente su propósito.

## Backend (ruta: `backend/app/tests`)
- Unit tests are also grouped under `backend/app/tests/unit`.
- `backend/app/tests/unit/test_unit_metrics.py`: Unit — validates `compute_metrics()`.
- `backend/app/tests/unit/test_unit_preprocessing.py`: Unit — preprocessing logic without DB.
- `backend/app/tests/unit/test_unit_auth.py`: Unit — auth hash/verify helpers.
- Existing integration/regression tests remain in `backend/app/tests/`.

## Frontend (ruta: `frontend/tests/unit`)

- `test_api.test.js`: Unit — mocks de `axios` y verificación de que `src/services/api.js` llama a los endpoints correctos.
- `test_settings.test.jsx`: Unit — mock de `src/services/api` y comprobación de que `Settings` rellena el formulario con los datos de la API al montar.

## Cómo ejecutar sólo las unitarias

- Backend (desde la raíz):

```powershell
python -m pytest backend/app/tests/test_metrics.py backend/app/tests/test_preprocessing.py backend/app/tests/test_auth.py -q
```

- Frontend (desde `frontend/`):

```bash
npm ci
npm run test
```

> Nota: Para ejecutar integraciones que usan DB en memoria no hacen falta servicios externos.

Si quieres, puedo:
- Añadir más tests unitarios para componentes React (Settings, Alerts) y helpers.
- Agrupar todos los tests unitarios en un único comando o script.

*** End Patch