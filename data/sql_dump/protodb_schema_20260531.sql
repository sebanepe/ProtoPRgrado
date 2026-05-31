--
-- PostgreSQL database dump
--

\restrict G6ZpHYhHM5Yxo8u9IawdbtRbDeSXdTu9aABHWRs0u73ptDb0o6gIE8D0XzCqpFF

-- Dumped from database version 15.15 (Debian 15.15-1.pgdg13+1)
-- Dumped by pg_dump version 15.15 (Debian 15.15-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alert_status_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.alert_status_history (
    id integer NOT NULL,
    alert_id integer NOT NULL,
    changed_by_id integer,
    old_status character varying(50),
    new_status character varying(50) NOT NULL,
    comment text,
    changed_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: alert_status_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.alert_status_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: alert_status_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.alert_status_history_id_seq OWNED BY public.alert_status_history.id;


--
-- Name: case_comments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.case_comments (
    id integer NOT NULL,
    case_id integer NOT NULL,
    user_id integer NOT NULL,
    comment text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: case_comments_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.case_comments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: case_comments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.case_comments_id_seq OWNED BY public.case_comments.id;


--
-- Name: datasets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.datasets (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    original_filename character varying(255),
    file_name character varying(255),
    file_path character varying(1024),
    file_hash character varying(255),
    source_type character varying(50) NOT NULL,
    total_records integer NOT NULL,
    valid_records integer NOT NULL,
    invalid_records integer NOT NULL,
    status character varying(50) NOT NULL,
    uploaded_by_id integer,
    metadata_json text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    processed_at timestamp with time zone,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    error_message text
);


--
-- Name: datasets_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.datasets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: datasets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.datasets_id_seq OWNED BY public.datasets.id;


--
-- Name: feature_sets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.feature_sets (
    id integer NOT NULL,
    dataset_id integer,
    preprocessing_run_id integer,
    name character varying(255) NOT NULL,
    file_path character varying(1024) NOT NULL,
    row_count integer NOT NULL,
    feature_columns_json text,
    excluded_columns_json text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    smote_report_json text,
    pipeline_path character varying(1024)
);


--
-- Name: feature_sets_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.feature_sets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: feature_sets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.feature_sets_id_seq OWNED BY public.feature_sets.id;


--
-- Name: fraud_alerts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fraud_alerts (
    id integer NOT NULL,
    scored_transaction_id integer,
    transaction_id integer,
    scoring_run_id integer,
    assigned_to_id integer,
    reviewed_by_id integer,
    model_name character varying(255),
    risk_score double precision NOT NULL,
    risk_level character varying(50) NOT NULL,
    status character varying(50) NOT NULL,
    notes text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: fraud_alerts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.fraud_alerts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: fraud_alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.fraud_alerts_id_seq OWNED BY public.fraud_alerts.id;


--
-- Name: fraud_cases; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fraud_cases (
    id integer NOT NULL,
    alert_id integer,
    assigned_to_id integer,
    opened_by_id integer,
    case_number character varying(100) NOT NULL,
    status character varying(50) NOT NULL,
    priority character varying(50) NOT NULL,
    summary text,
    conclusion text,
    opened_at timestamp with time zone DEFAULT now() NOT NULL,
    closed_at timestamp with time zone
);


--
-- Name: fraud_cases_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.fraud_cases_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: fraud_cases_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.fraud_cases_id_seq OWNED BY public.fraud_cases.id;


--
-- Name: ml_models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ml_models (
    id integer NOT NULL,
    feature_set_id integer,
    trained_by_id integer,
    name character varying(255) NOT NULL,
    algorithm character varying(255) NOT NULL,
    version character varying(100) NOT NULL,
    artifact_path character varying(1024) NOT NULL,
    target_column character varying(255) NOT NULL,
    feature_columns_json text,
    excluded_columns_json text,
    hyperparameters_json text,
    metrics_json text,
    accuracy double precision,
    precision_score double precision,
    recall_score double precision,
    f1_score double precision,
    roc_auc double precision,
    confusion_matrix_json text,
    status character varying(50) NOT NULL,
    is_candidate boolean NOT NULL,
    trained_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: ml_models_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ml_models_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ml_models_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ml_models_id_seq OWNED BY public.ml_models.id;


--
-- Name: model_config; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_config (
    id integer NOT NULL,
    active_model_id integer,
    created_by_id integer,
    alert_threshold double precision NOT NULL,
    is_active boolean NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    deactivated_at timestamp with time zone,
    updated_by character varying(255),
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: model_config_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.model_config_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_config_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.model_config_id_seq OWNED BY public.model_config.id;


--
-- Name: model_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_results (
    id integer NOT NULL,
    model_name character varying(255) NOT NULL,
    version character varying(50),
    accuracy double precision,
    "precision" double precision,
    recall double precision,
    f1_score double precision,
    roc_auc double precision,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    is_active boolean NOT NULL
);


--
-- Name: model_results_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.model_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.model_results_id_seq OWNED BY public.model_results.id;


--
-- Name: permissions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.permissions (
    id integer NOT NULL,
    code character varying(200) NOT NULL,
    module character varying(100) NOT NULL,
    action character varying(100) NOT NULL,
    description text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: permissions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.permissions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: permissions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.permissions_id_seq OWNED BY public.permissions.id;


--
-- Name: preprocessing_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.preprocessing_runs (
    id integer NOT NULL,
    input_dataset_id integer,
    executed_by_id integer,
    output_file_path character varying(1024),
    status character varying(50) NOT NULL,
    total_records integer NOT NULL,
    processed_records integer NOT NULL,
    removed_records integer NOT NULL,
    params_json text,
    error_message text,
    started_at timestamp with time zone,
    finished_at timestamp with time zone
);


--
-- Name: preprocessing_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.preprocessing_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: preprocessing_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.preprocessing_runs_id_seq OWNED BY public.preprocessing_runs.id;


--
-- Name: report_exports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.report_exports (
    id integer NOT NULL,
    requested_by_id integer,
    report_type character varying(100) NOT NULL,
    file_path character varying(1024),
    status character varying(50) NOT NULL,
    filters_json text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone
);


--
-- Name: report_exports_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.report_exports_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: report_exports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.report_exports_id_seq OWNED BY public.report_exports.id;


--
-- Name: role_permissions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.role_permissions (
    id integer NOT NULL,
    role_id integer NOT NULL,
    permission_id integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: role_permissions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.role_permissions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: role_permissions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.role_permissions_id_seq OWNED BY public.role_permissions.id;


--
-- Name: roles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.roles (
    id integer NOT NULL,
    code character varying(100) NOT NULL,
    name character varying(255) NOT NULL,
    description text,
    is_system boolean NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: roles_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.roles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: roles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.roles_id_seq OWNED BY public.roles.id;


--
-- Name: rule_alert_reviews; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.rule_alert_reviews (
    id integer NOT NULL,
    source_run character varying(255) NOT NULL,
    alert_id character varying(255),
    summary_alert_id character varying(255),
    rule_code character varying(255) NOT NULL,
    transaction_id character varying(255),
    customer_hash character varying(255),
    previous_status character varying(50),
    new_status character varying(50) NOT NULL,
    analyst_label character varying(50),
    analyst_notes text,
    reviewed_by_id integer,
    reviewed_at timestamp with time zone DEFAULT now() NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: rule_alert_reviews_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.rule_alert_reviews_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: rule_alert_reviews_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.rule_alert_reviews_id_seq OWNED BY public.rule_alert_reviews.id;


--
-- Name: scored_transactions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scored_transactions (
    id integer NOT NULL,
    scoring_run_id integer NOT NULL,
    transaction_id integer,
    risk_score double precision NOT NULL,
    risk_level character varying(50) NOT NULL,
    prediction_label integer,
    explanation_json text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: scored_transactions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scored_transactions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: scored_transactions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scored_transactions_id_seq OWNED BY public.scored_transactions.id;


--
-- Name: scoring_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scoring_runs (
    id integer NOT NULL,
    model_id integer,
    dataset_id integer,
    feature_set_id integer,
    model_config_id integer,
    executed_by_id integer,
    threshold_used double precision,
    total_scored integer NOT NULL,
    alerts_generated integer NOT NULL,
    status character varying(50) NOT NULL,
    error_message text,
    started_at timestamp with time zone,
    finished_at timestamp with time zone
);


--
-- Name: scoring_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scoring_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: scoring_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scoring_runs_id_seq OWNED BY public.scoring_runs.id;


--
-- Name: system_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.system_logs (
    id integer NOT NULL,
    action character varying(100) NOT NULL,
    description text,
    user_id integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    ip character varying(64),
    user_agent character varying(1024)
);


--
-- Name: system_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.system_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: system_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.system_logs_id_seq OWNED BY public.system_logs.id;


--
-- Name: transactions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.transactions (
    id integer NOT NULL,
    transaction_id character varying(100) NOT NULL,
    amount numeric(14,2) NOT NULL,
    transaction_type character varying(50),
    channel character varying(50),
    location character varying(255),
    device_id character varying(255),
    customer_hash character varying(255),
    transaction_datetime timestamp with time zone NOT NULL,
    is_fraud boolean NOT NULL,
    imported_at timestamp with time zone DEFAULT now() NOT NULL,
    dataset_id integer,
    merchant_hash character varying(255),
    merchant_code character varying(100),
    terminal_code character varying(100),
    merchant_name character varying(255),
    country_code character varying(10),
    pos_entry_mode integer,
    has_pinblock integer,
    card_brand character varying(50),
    merchant_rubro_proxy character varying(20)
);


--
-- Name: transactions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.transactions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: transactions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.transactions_id_seq OWNED BY public.transactions.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id integer NOT NULL,
    full_name character varying(255) NOT NULL,
    email character varying(255) NOT NULL,
    password_hash character varying(255) NOT NULL,
    role character varying(50) NOT NULL,
    role_id integer,
    is_active boolean NOT NULL,
    failed_login_attempts integer NOT NULL,
    locked_until timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: alert_status_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alert_status_history ALTER COLUMN id SET DEFAULT nextval('public.alert_status_history_id_seq'::regclass);


--
-- Name: case_comments id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_comments ALTER COLUMN id SET DEFAULT nextval('public.case_comments_id_seq'::regclass);


--
-- Name: datasets id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets ALTER COLUMN id SET DEFAULT nextval('public.datasets_id_seq'::regclass);


--
-- Name: feature_sets id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.feature_sets ALTER COLUMN id SET DEFAULT nextval('public.feature_sets_id_seq'::regclass);


--
-- Name: fraud_alerts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_alerts ALTER COLUMN id SET DEFAULT nextval('public.fraud_alerts_id_seq'::regclass);


--
-- Name: fraud_cases id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_cases ALTER COLUMN id SET DEFAULT nextval('public.fraud_cases_id_seq'::regclass);


--
-- Name: ml_models id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ml_models ALTER COLUMN id SET DEFAULT nextval('public.ml_models_id_seq'::regclass);


--
-- Name: model_config id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_config ALTER COLUMN id SET DEFAULT nextval('public.model_config_id_seq'::regclass);


--
-- Name: model_results id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_results ALTER COLUMN id SET DEFAULT nextval('public.model_results_id_seq'::regclass);


--
-- Name: permissions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.permissions ALTER COLUMN id SET DEFAULT nextval('public.permissions_id_seq'::regclass);


--
-- Name: preprocessing_runs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.preprocessing_runs ALTER COLUMN id SET DEFAULT nextval('public.preprocessing_runs_id_seq'::regclass);


--
-- Name: report_exports id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.report_exports ALTER COLUMN id SET DEFAULT nextval('public.report_exports_id_seq'::regclass);


--
-- Name: role_permissions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.role_permissions ALTER COLUMN id SET DEFAULT nextval('public.role_permissions_id_seq'::regclass);


--
-- Name: roles id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.roles ALTER COLUMN id SET DEFAULT nextval('public.roles_id_seq'::regclass);


--
-- Name: rule_alert_reviews id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_alert_reviews ALTER COLUMN id SET DEFAULT nextval('public.rule_alert_reviews_id_seq'::regclass);


--
-- Name: scored_transactions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scored_transactions ALTER COLUMN id SET DEFAULT nextval('public.scored_transactions_id_seq'::regclass);


--
-- Name: scoring_runs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_runs ALTER COLUMN id SET DEFAULT nextval('public.scoring_runs_id_seq'::regclass);


--
-- Name: system_logs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_logs ALTER COLUMN id SET DEFAULT nextval('public.system_logs_id_seq'::regclass);


--
-- Name: transactions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions ALTER COLUMN id SET DEFAULT nextval('public.transactions_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: alert_status_history alert_status_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alert_status_history
    ADD CONSTRAINT alert_status_history_pkey PRIMARY KEY (id);


--
-- Name: case_comments case_comments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_comments
    ADD CONSTRAINT case_comments_pkey PRIMARY KEY (id);


--
-- Name: datasets datasets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_pkey PRIMARY KEY (id);


--
-- Name: feature_sets feature_sets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.feature_sets
    ADD CONSTRAINT feature_sets_pkey PRIMARY KEY (id);


--
-- Name: fraud_alerts fraud_alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_alerts
    ADD CONSTRAINT fraud_alerts_pkey PRIMARY KEY (id);


--
-- Name: fraud_cases fraud_cases_case_number_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_cases
    ADD CONSTRAINT fraud_cases_case_number_key UNIQUE (case_number);


--
-- Name: fraud_cases fraud_cases_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_cases
    ADD CONSTRAINT fraud_cases_pkey PRIMARY KEY (id);


--
-- Name: ml_models ml_models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ml_models
    ADD CONSTRAINT ml_models_pkey PRIMARY KEY (id);


--
-- Name: model_config model_config_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_config
    ADD CONSTRAINT model_config_pkey PRIMARY KEY (id);


--
-- Name: model_results model_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_results
    ADD CONSTRAINT model_results_pkey PRIMARY KEY (id);


--
-- Name: permissions permissions_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.permissions
    ADD CONSTRAINT permissions_code_key UNIQUE (code);


--
-- Name: permissions permissions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.permissions
    ADD CONSTRAINT permissions_pkey PRIMARY KEY (id);


--
-- Name: preprocessing_runs preprocessing_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.preprocessing_runs
    ADD CONSTRAINT preprocessing_runs_pkey PRIMARY KEY (id);


--
-- Name: report_exports report_exports_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.report_exports
    ADD CONSTRAINT report_exports_pkey PRIMARY KEY (id);


--
-- Name: role_permissions role_permissions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.role_permissions
    ADD CONSTRAINT role_permissions_pkey PRIMARY KEY (id);


--
-- Name: roles roles_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.roles
    ADD CONSTRAINT roles_code_key UNIQUE (code);


--
-- Name: roles roles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.roles
    ADD CONSTRAINT roles_pkey PRIMARY KEY (id);


--
-- Name: rule_alert_reviews rule_alert_reviews_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_alert_reviews
    ADD CONSTRAINT rule_alert_reviews_pkey PRIMARY KEY (id);


--
-- Name: scored_transactions scored_transactions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scored_transactions
    ADD CONSTRAINT scored_transactions_pkey PRIMARY KEY (id);


--
-- Name: scoring_runs scoring_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_runs
    ADD CONSTRAINT scoring_runs_pkey PRIMARY KEY (id);


--
-- Name: system_logs system_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_logs
    ADD CONSTRAINT system_logs_pkey PRIMARY KEY (id);


--
-- Name: transactions transactions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: ix_alert_status_history_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_alert_status_history_id ON public.alert_status_history USING btree (id);


--
-- Name: ix_case_comments_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_case_comments_id ON public.case_comments USING btree (id);


--
-- Name: ix_datasets_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_datasets_id ON public.datasets USING btree (id);


--
-- Name: ix_feature_sets_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_feature_sets_id ON public.feature_sets USING btree (id);


--
-- Name: ix_fraud_alerts_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_fraud_alerts_id ON public.fraud_alerts USING btree (id);


--
-- Name: ix_fraud_cases_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_fraud_cases_id ON public.fraud_cases USING btree (id);


--
-- Name: ix_ml_models_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_ml_models_id ON public.ml_models USING btree (id);


--
-- Name: ix_model_config_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_model_config_id ON public.model_config USING btree (id);


--
-- Name: ix_model_results_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_model_results_id ON public.model_results USING btree (id);


--
-- Name: ix_permissions_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_permissions_id ON public.permissions USING btree (id);


--
-- Name: ix_preprocessing_runs_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_preprocessing_runs_id ON public.preprocessing_runs USING btree (id);


--
-- Name: ix_report_exports_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_report_exports_id ON public.report_exports USING btree (id);


--
-- Name: ix_role_permissions_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_role_permissions_id ON public.role_permissions USING btree (id);


--
-- Name: ix_roles_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_roles_id ON public.roles USING btree (id);


--
-- Name: ix_rule_alert_review_lookup; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_review_lookup ON public.rule_alert_reviews USING btree (source_run, alert_id, summary_alert_id);


--
-- Name: ix_rule_alert_review_rule; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_review_rule ON public.rule_alert_reviews USING btree (source_run, rule_code);


--
-- Name: ix_rule_alert_review_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_review_status ON public.rule_alert_reviews USING btree (source_run, new_status);


--
-- Name: ix_rule_alert_reviews_alert_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_reviews_alert_id ON public.rule_alert_reviews USING btree (alert_id);


--
-- Name: ix_rule_alert_reviews_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_reviews_id ON public.rule_alert_reviews USING btree (id);


--
-- Name: ix_rule_alert_reviews_new_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_reviews_new_status ON public.rule_alert_reviews USING btree (new_status);


--
-- Name: ix_rule_alert_reviews_rule_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_reviews_rule_code ON public.rule_alert_reviews USING btree (rule_code);


--
-- Name: ix_rule_alert_reviews_source_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_reviews_source_run ON public.rule_alert_reviews USING btree (source_run);


--
-- Name: ix_rule_alert_reviews_summary_alert_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_rule_alert_reviews_summary_alert_id ON public.rule_alert_reviews USING btree (summary_alert_id);


--
-- Name: ix_scored_transactions_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_scored_transactions_id ON public.scored_transactions USING btree (id);


--
-- Name: ix_scoring_runs_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_scoring_runs_id ON public.scoring_runs USING btree (id);


--
-- Name: ix_system_logs_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_system_logs_id ON public.system_logs USING btree (id);


--
-- Name: ix_transactions_customer_hash; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_transactions_customer_hash ON public.transactions USING btree (customer_hash);


--
-- Name: ix_transactions_dataset_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_transactions_dataset_id ON public.transactions USING btree (dataset_id);


--
-- Name: ix_transactions_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_transactions_id ON public.transactions USING btree (id);


--
-- Name: ix_users_email; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ix_users_email ON public.users USING btree (email);


--
-- Name: ix_users_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_users_id ON public.users USING btree (id);


--
-- Name: uq_model_name_version; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_model_name_version ON public.ml_models USING btree (name, version);


--
-- Name: uq_role_permission; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_role_permission ON public.role_permissions USING btree (role_id, permission_id);


--
-- Name: uq_transactions_transaction_dataset; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_transactions_transaction_dataset ON public.transactions USING btree (transaction_id, dataset_id);


--
-- Name: alert_status_history alert_status_history_alert_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alert_status_history
    ADD CONSTRAINT alert_status_history_alert_id_fkey FOREIGN KEY (alert_id) REFERENCES public.fraud_alerts(id) ON DELETE CASCADE;


--
-- Name: alert_status_history alert_status_history_changed_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alert_status_history
    ADD CONSTRAINT alert_status_history_changed_by_id_fkey FOREIGN KEY (changed_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: case_comments case_comments_case_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_comments
    ADD CONSTRAINT case_comments_case_id_fkey FOREIGN KEY (case_id) REFERENCES public.fraud_cases(id) ON DELETE CASCADE;


--
-- Name: case_comments case_comments_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_comments
    ADD CONSTRAINT case_comments_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: datasets datasets_uploaded_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_uploaded_by_id_fkey FOREIGN KEY (uploaded_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: feature_sets feature_sets_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.feature_sets
    ADD CONSTRAINT feature_sets_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON DELETE SET NULL;


--
-- Name: feature_sets feature_sets_preprocessing_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.feature_sets
    ADD CONSTRAINT feature_sets_preprocessing_run_id_fkey FOREIGN KEY (preprocessing_run_id) REFERENCES public.preprocessing_runs(id) ON DELETE SET NULL;


--
-- Name: transactions fk_transactions_dataset_id; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT fk_transactions_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON DELETE SET NULL;


--
-- Name: fraud_alerts fraud_alerts_assigned_to_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_alerts
    ADD CONSTRAINT fraud_alerts_assigned_to_id_fkey FOREIGN KEY (assigned_to_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: fraud_alerts fraud_alerts_reviewed_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_alerts
    ADD CONSTRAINT fraud_alerts_reviewed_by_id_fkey FOREIGN KEY (reviewed_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: fraud_alerts fraud_alerts_scored_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_alerts
    ADD CONSTRAINT fraud_alerts_scored_transaction_id_fkey FOREIGN KEY (scored_transaction_id) REFERENCES public.scored_transactions(id) ON DELETE SET NULL;


--
-- Name: fraud_alerts fraud_alerts_scoring_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_alerts
    ADD CONSTRAINT fraud_alerts_scoring_run_id_fkey FOREIGN KEY (scoring_run_id) REFERENCES public.scoring_runs(id) ON DELETE SET NULL;


--
-- Name: fraud_alerts fraud_alerts_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_alerts
    ADD CONSTRAINT fraud_alerts_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id) ON DELETE SET NULL;


--
-- Name: fraud_cases fraud_cases_alert_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_cases
    ADD CONSTRAINT fraud_cases_alert_id_fkey FOREIGN KEY (alert_id) REFERENCES public.fraud_alerts(id) ON DELETE SET NULL;


--
-- Name: fraud_cases fraud_cases_assigned_to_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_cases
    ADD CONSTRAINT fraud_cases_assigned_to_id_fkey FOREIGN KEY (assigned_to_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: fraud_cases fraud_cases_opened_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fraud_cases
    ADD CONSTRAINT fraud_cases_opened_by_id_fkey FOREIGN KEY (opened_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: ml_models ml_models_feature_set_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ml_models
    ADD CONSTRAINT ml_models_feature_set_id_fkey FOREIGN KEY (feature_set_id) REFERENCES public.feature_sets(id) ON DELETE SET NULL;


--
-- Name: ml_models ml_models_trained_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ml_models
    ADD CONSTRAINT ml_models_trained_by_id_fkey FOREIGN KEY (trained_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: model_config model_config_active_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_config
    ADD CONSTRAINT model_config_active_model_id_fkey FOREIGN KEY (active_model_id) REFERENCES public.ml_models(id) ON DELETE SET NULL;


--
-- Name: model_config model_config_created_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_config
    ADD CONSTRAINT model_config_created_by_id_fkey FOREIGN KEY (created_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: preprocessing_runs preprocessing_runs_executed_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.preprocessing_runs
    ADD CONSTRAINT preprocessing_runs_executed_by_id_fkey FOREIGN KEY (executed_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: preprocessing_runs preprocessing_runs_input_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.preprocessing_runs
    ADD CONSTRAINT preprocessing_runs_input_dataset_id_fkey FOREIGN KEY (input_dataset_id) REFERENCES public.datasets(id) ON DELETE SET NULL;


--
-- Name: report_exports report_exports_requested_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.report_exports
    ADD CONSTRAINT report_exports_requested_by_id_fkey FOREIGN KEY (requested_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: role_permissions role_permissions_permission_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.role_permissions
    ADD CONSTRAINT role_permissions_permission_id_fkey FOREIGN KEY (permission_id) REFERENCES public.permissions(id) ON DELETE CASCADE;


--
-- Name: role_permissions role_permissions_role_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.role_permissions
    ADD CONSTRAINT role_permissions_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id) ON DELETE CASCADE;


--
-- Name: rule_alert_reviews rule_alert_reviews_reviewed_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_alert_reviews
    ADD CONSTRAINT rule_alert_reviews_reviewed_by_id_fkey FOREIGN KEY (reviewed_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: scored_transactions scored_transactions_scoring_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scored_transactions
    ADD CONSTRAINT scored_transactions_scoring_run_id_fkey FOREIGN KEY (scoring_run_id) REFERENCES public.scoring_runs(id) ON DELETE CASCADE;


--
-- Name: scored_transactions scored_transactions_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scored_transactions
    ADD CONSTRAINT scored_transactions_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id) ON DELETE SET NULL;


--
-- Name: scoring_runs scoring_runs_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_runs
    ADD CONSTRAINT scoring_runs_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON DELETE SET NULL;


--
-- Name: scoring_runs scoring_runs_executed_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_runs
    ADD CONSTRAINT scoring_runs_executed_by_id_fkey FOREIGN KEY (executed_by_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: scoring_runs scoring_runs_feature_set_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_runs
    ADD CONSTRAINT scoring_runs_feature_set_id_fkey FOREIGN KEY (feature_set_id) REFERENCES public.feature_sets(id) ON DELETE SET NULL;


--
-- Name: scoring_runs scoring_runs_model_config_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_runs
    ADD CONSTRAINT scoring_runs_model_config_id_fkey FOREIGN KEY (model_config_id) REFERENCES public.model_config(id) ON DELETE SET NULL;


--
-- Name: scoring_runs scoring_runs_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_runs
    ADD CONSTRAINT scoring_runs_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.ml_models(id) ON DELETE SET NULL;


--
-- Name: system_logs system_logs_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_logs
    ADD CONSTRAINT system_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: users users_role_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id) ON DELETE SET NULL;


--
-- PostgreSQL database dump complete
--

\unrestrict G6ZpHYhHM5Yxo8u9IawdbtRbDeSXdTu9aABHWRs0u73ptDb0o6gIE8D0XzCqpFF

