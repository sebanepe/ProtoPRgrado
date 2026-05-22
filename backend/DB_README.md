Database initialization

- For local development we use SQLite by default. Set `DATABASE_URL=sqlite:///./backend/dev.db` in `.env` or use `.env.example`.
- To create tables and seed roles/permissions/admin run:

  python -m backend.app.init_db

- For production, create a Postgres database and set `DATABASE_URL` accordingly. Use `backend/create_database.sql` as a starting point.

Security note: the default admin password in `.env.example` is for local/dev use only. In production use a secrets manager or strong random password.
