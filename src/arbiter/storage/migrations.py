"""Ordered, append-only DuckDB schema migrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    """One transactionally applied schema change."""

    version: int
    name: str
    sql: str


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="metadata_foundation",
        sql="""
            CREATE TABLE metadata_sync_runs (
                run_id VARCHAR PRIMARY KEY,
                started_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                status VARCHAR NOT NULL,
                market_count INTEGER NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL DEFAULT 0,
                series_count INTEGER NOT NULL DEFAULT 0,
                error VARCHAR
            );

            CREATE TABLE markets (
                ticker VARCHAR PRIMARY KEY,
                event_ticker VARCHAR NOT NULL,
                series_ticker VARCHAR,
                market_type VARCHAR,
                title VARCHAR NOT NULL,
                subtitle VARCHAR,
                yes_sub_title VARCHAR,
                no_sub_title VARCHAR,
                status VARCHAR NOT NULL,
                created_time TIMESTAMPTZ,
                market_updated_time TIMESTAMPTZ,
                open_time TIMESTAMPTZ,
                close_time TIMESTAMPTZ,
                expiration_time TIMESTAMPTZ,
                latest_expiration_time TIMESTAMPTZ,
                expected_expiration_time TIMESTAMPTZ,
                settlement_ts TIMESTAMPTZ,
                occurrence_datetime TIMESTAMPTZ,
                strike_type VARCHAR,
                floor_strike DECIMAL(38, 18),
                cap_strike DECIMAL(38, 18),
                functional_strike VARCHAR,
                custom_strike_json JSON,
                rules_primary VARCHAR,
                rules_secondary VARCHAR,
                early_close_condition VARCHAR,
                price_level_structure VARCHAR,
                price_ranges_json JSON NOT NULL,
                result VARCHAR,
                raw_json JSON NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );

            CREATE TABLE events (
                ticker VARCHAR PRIMARY KEY,
                series_ticker VARCHAR,
                title VARCHAR NOT NULL,
                subtitle VARCHAR,
                category VARCHAR,
                mutually_exclusive BOOLEAN,
                available_on_brokers BOOLEAN,
                market_tickers_json JSON NOT NULL,
                last_updated_ts TIMESTAMPTZ,
                raw_json JSON NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );

            CREATE TABLE series (
                ticker VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                frequency VARCHAR,
                category VARCHAR,
                tags_json JSON NOT NULL,
                fee_type VARCHAR,
                fee_multiplier DECIMAL(38, 18),
                settlement_sources_json JSON NOT NULL,
                contract_url VARCHAR,
                contract_terms_url VARCHAR,
                last_updated_ts TIMESTAMPTZ,
                raw_json JSON NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
        """,
    ),
    Migration(
        version=2,
        name="trusted_relations",
        sql="""
            CREATE TABLE relations (
                relation_id VARCHAR PRIMARY KEY,
                relation_type VARCHAR NOT NULL,
                market_tickers_json JSON NOT NULL,
                antecedent VARCHAR,
                consequent VARCHAR,
                source VARCHAR NOT NULL,
                confidence DOUBLE,
                verified BOOLEAN NOT NULL,
                rationale VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
        """,
    ),
    Migration(
        version=3,
        name="fee_schedules_and_opportunities",
        sql="""
            ALTER TABLE events
            ADD COLUMN fee_changes_json JSON;

            UPDATE events
            SET fee_changes_json = '[]'
            WHERE fee_changes_json IS NULL;

            CREATE TABLE opportunities (
                opportunity_id VARCHAR PRIMARY KEY,
                component_id VARCHAR NOT NULL,
                detected_at TIMESTAMPTZ NOT NULL,
                ended_at TIMESTAMPTZ,
                stage VARCHAR NOT NULL,
                relation_types_json JSON NOT NULL,
                num_markets INTEGER NOT NULL,
                num_legs INTEGER NOT NULL,
                capital_required DECIMAL(38, 18),
                gross_profit DECIMAL(38, 18),
                fees DECIMAL(38, 18),
                net_profit DECIMAL(38, 18),
                gross_edge DECIMAL(38, 18),
                net_edge DECIMAL(38, 18),
                fee_policy_version VARCHAR,
                fee_policies_json JSON NOT NULL DEFAULT '[]',
                paper_execution_status VARCHAR NOT NULL,
                paper_execution_reason VARCHAR,
                metadata_json JSON NOT NULL
            );

            CREATE TABLE portfolio_legs (
                opportunity_id VARCHAR NOT NULL,
                ticker VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                price DECIMAL(38, 18) NOT NULL,
                quantity DECIMAL(38, 18) NOT NULL,
                source_side VARCHAR NOT NULL,
                source_price DECIMAL(38, 18) NOT NULL,
                fee DECIMAL(38, 18)
            );
        """,
    ),
    Migration(
        version=4,
        name="scanner_runs_and_observations",
        sql="""
            ALTER TABLE events
            ADD COLUMN fee_type_override VARCHAR;

            ALTER TABLE events
            ADD COLUMN fee_multiplier_override DECIMAL(38, 18);

            ALTER TABLE opportunities
            ADD COLUMN run_id VARCHAR;

            ALTER TABLE opportunities
            ADD COLUMN opened_at TIMESTAMPTZ;

            ALTER TABLE opportunities
            ADD COLUMN closed_at TIMESTAMPTZ;

            ALTER TABLE opportunities
            ADD COLUMN updated_at TIMESTAMPTZ;

            CREATE TABLE run_manifests (
                run_id VARCHAR PRIMARY KEY,
                run_type VARCHAR NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                ended_at TIMESTAMPTZ,
                status VARCHAR NOT NULL,
                schema_version INTEGER NOT NULL,
                config_hash VARCHAR,
                metadata_hash VARCHAR,
                relations_hash VARCHAR,
                fee_policy_hash VARCHAR,
                metadata_json JSON NOT NULL DEFAULT '{}',
                error VARCHAR
            );

            CREATE TABLE market_observation_windows (
                observation_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                market_ticker VARCHAR NOT NULL,
                opened_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                closed_at TIMESTAMPTZ,
                opened_event_index BIGINT NOT NULL,
                last_event_index BIGINT NOT NULL,
                closed_event_index BIGINT,
                status VARCHAR NOT NULL,
                start_sequence BIGINT,
                end_sequence BIGINT,
                connection_id VARCHAR,
                stale_reason VARCHAR,
                resync_reason VARCHAR,
                metadata_json JSON NOT NULL DEFAULT '{}'
            );

            CREATE TABLE opportunity_observations (
                observation_id VARCHAR PRIMARY KEY,
                opportunity_id VARCHAR,
                run_id VARCHAR NOT NULL,
                component_id VARCHAR NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL,
                event_index BIGINT NOT NULL,
                transition VARCHAR NOT NULL,
                stage VARCHAR,
                solver_status VARCHAR,
                reason VARCHAR,
                solve_duration_ms DOUBLE NOT NULL,
                num_states INTEGER NOT NULL,
                num_instruments INTEGER NOT NULL,
                num_legs INTEGER NOT NULL,
                capital_required DECIMAL(38, 18),
                gross_profit DECIMAL(38, 18),
                fees DECIMAL(38, 18),
                net_profit DECIMAL(38, 18),
                gross_edge DECIMAL(38, 18),
                net_edge DECIMAL(38, 18),
                capacity DECIMAL(38, 18),
                fee_policy_version VARCHAR,
                fee_policies_json JSON NOT NULL DEFAULT '[]',
                evidence_json JSON NOT NULL DEFAULT '{}',
                metadata_json JSON NOT NULL DEFAULT '{}'
            );
        """,
    ),
    Migration(
        version=5,
        name="replay_paper_evidence",
        sql="""
            ALTER TABLE run_manifests
            ADD COLUMN manifest_version INTEGER;

            UPDATE run_manifests
            SET manifest_version = 1
            WHERE manifest_version IS NULL;

            ALTER TABLE run_manifests
            ADD COLUMN recording_format_version INTEGER;

            UPDATE run_manifests
            SET recording_format_version = 1
            WHERE recording_format_version IS NULL;

            ALTER TABLE run_manifests
            ADD COLUMN event_schema_version INTEGER;

            UPDATE run_manifests
            SET event_schema_version = 1
            WHERE event_schema_version IS NULL;

            ALTER TABLE run_manifests
            ADD COLUMN recording_id VARCHAR;

            ALTER TABLE run_manifests
            ADD COLUMN source_run_id VARCHAR;

            ALTER TABLE run_manifests
            ADD COLUMN first_event_index BIGINT;

            ALTER TABLE run_manifests
            ADD COLUMN last_event_index BIGINT;

            ALTER TABLE run_manifests
            ADD COLUMN event_count BIGINT;

            UPDATE run_manifests
            SET event_count = 0
            WHERE event_count IS NULL;

            ALTER TABLE run_manifests
            ADD COLUMN event_stream_hash VARCHAR;

            ALTER TABLE run_manifests
            ADD COLUMN input_payload_json JSON;

            UPDATE run_manifests
            SET input_payload_json = '{}'
            WHERE input_payload_json IS NULL;

            ALTER TABLE opportunities
            ADD COLUMN censored_at TIMESTAMPTZ;

            ALTER TABLE opportunities
            ADD COLUMN censored_event_index BIGINT;

            ALTER TABLE opportunities
            ADD COLUMN censor_reason VARCHAR;

            ALTER TABLE opportunities
            ADD COLUMN paper_execution_attempt_id VARCHAR;

            ALTER TABLE opportunities
            ADD COLUMN paper_execution_source_event_index BIGINT;

            CREATE TABLE opportunity_observation_legs (
                observation_id VARCHAR NOT NULL,
                leg_index INTEGER NOT NULL,
                opportunity_id VARCHAR NOT NULL,
                ticker VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                price DECIMAL(38, 18) NOT NULL,
                quantity DECIMAL(38, 18) NOT NULL,
                source_side VARCHAR NOT NULL,
                source_price DECIMAL(38, 18) NOT NULL,
                fee DECIMAL(38, 18),
                PRIMARY KEY (observation_id, leg_index)
            );

            CREATE TABLE paper_executions (
                attempt_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                opportunity_id VARCHAR NOT NULL,
                source_observation_id VARCHAR NOT NULL,
                source_event_index BIGINT NOT NULL,
                detected_at TIMESTAMPTZ NOT NULL,
                scheduled_at TIMESTAMPTZ NOT NULL,
                attempted_at TIMESTAMPTZ,
                resolved_at TIMESTAMPTZ NOT NULL,
                latency_ms BIGINT NOT NULL,
                status VARCHAR NOT NULL,
                failure_reason VARCHAR,
                minimum_terminal_payout DECIMAL(38, 18) NOT NULL,
                expected_profit DECIMAL(38, 18) NOT NULL,
                simulated_locked_profit DECIMAL(38, 18),
                expected_cost DECIMAL(38, 18) NOT NULL,
                actual_cost DECIMAL(38, 18),
                expected_fees DECIMAL(38, 18) NOT NULL,
                actual_fees DECIMAL(38, 18),
                expected_fee_policy_version VARCHAR,
                expected_fee_policies_json JSON NOT NULL,
                expected_fee_quotes_json JSON NOT NULL,
                execution_fee_policy_version VARCHAR,
                execution_fee_policies_json JSON NOT NULL,
                execution_fee_quotes_json JSON NOT NULL,
                evidence_json JSON NOT NULL DEFAULT '{}',
                payload_hash VARCHAR NOT NULL
            );

            CREATE TABLE paper_execution_legs (
                attempt_id VARCHAR NOT NULL,
                leg_index INTEGER NOT NULL,
                ticker VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                quantity DECIMAL(38, 18) NOT NULL,
                expected_prices_json JSON NOT NULL,
                actual_prices_json JSON NOT NULL,
                expected_average_price DECIMAL(38, 18) NOT NULL,
                actual_average_price DECIMAL(38, 18),
                expected_cost DECIMAL(38, 18) NOT NULL,
                actual_cost DECIMAL(38, 18),
                fill_status VARCHAR NOT NULL,
                failure_reason VARCHAR,
                PRIMARY KEY (attempt_id, leg_index)
            );
        """,
    ),
    Migration(
        version=6,
        name="semantic_discovery_review",
        sql="""
            ALTER TABLE relations
            ADD COLUMN semantic_suggestion_id VARCHAR;

            ALTER TABLE relations
            ADD COLUMN semantic_rules_hash VARCHAR;

            ALTER TABLE relations
            ADD COLUMN semantic_timing_hash VARCHAR;

            CREATE TABLE semantic_embeddings (
                embedding_id VARCHAR PRIMARY KEY,
                market_ticker VARCHAR NOT NULL,
                canonical_text VARCHAR NOT NULL,
                canonical_text_hash VARCHAR NOT NULL,
                rules_hash VARCHAR NOT NULL,
                timing_hash VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json JSON NOT NULL,
                payload_hash VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                UNIQUE (market_ticker, canonical_text_hash, provider, model)
            );

            CREATE TABLE semantic_suggestions (
                suggestion_id VARCHAR PRIMARY KEY,
                market_a_ticker VARCHAR NOT NULL,
                market_b_ticker VARCHAR NOT NULL,
                market_a_text_hash VARCHAR NOT NULL,
                market_b_text_hash VARCHAR NOT NULL,
                market_a_rules_hash VARCHAR NOT NULL,
                market_b_rules_hash VARCHAR NOT NULL,
                market_a_timing_hash VARCHAR NOT NULL,
                market_b_timing_hash VARCHAR NOT NULL,
                market_a_title VARCHAR NOT NULL,
                market_b_title VARCHAR NOT NULL,
                market_a_rules_text VARCHAR NOT NULL,
                market_b_rules_text VARCHAR NOT NULL,
                market_a_timing_text VARCHAR NOT NULL,
                market_b_timing_text VARCHAR NOT NULL,
                embedding_provider VARCHAR NOT NULL,
                embedding_model VARCHAR NOT NULL,
                cosine_similarity DOUBLE NOT NULL,
                classifier_provider VARCHAR NOT NULL,
                classifier_model VARCHAR NOT NULL,
                prompt_version VARCHAR NOT NULL,
                prompt VARCHAR NOT NULL,
                raw_response VARCHAR NOT NULL,
                parsed_proposal_json JSON NOT NULL,
                relation VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL,
                rationale VARCHAR NOT NULL,
                requires_review BOOLEAN NOT NULL,
                review_state VARCHAR NOT NULL,
                reviewed_at TIMESTAMPTZ,
                approved_relation_id VARCHAR,
                stale_reason VARCHAR,
                payload_hash VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                CHECK (market_a_ticker < market_b_ticker),
                CHECK (cosine_similarity >= -1.0 AND cosine_similarity <= 1.0),
                CHECK (confidence >= 0.0 AND confidence <= 1.0),
                CHECK (requires_review),
                CHECK (
                    relation IN (
                        'A_IMPLIES_B', 'B_IMPLIES_A', 'EQUIVALENT',
                        'MUTUALLY_EXCLUSIVE', 'NONE', 'UNCERTAIN'
                    )
                ),
                CHECK (review_state IN ('pending', 'approved', 'rejected', 'uncertain', 'stale'))
            );

            CREATE INDEX semantic_suggestions_review_queue
            ON semantic_suggestions (review_state, created_at);
        """,
    ),
    Migration(
        version=7,
        name="research_observation_context",
        sql="""
            ALTER TABLE opportunity_observations
            ADD COLUMN market_tickers_json JSON;

            ALTER TABLE opportunity_observations
            ADD COLUMN relation_types_json JSON;

            ALTER TABLE opportunity_observations
            ADD COLUMN relation_sources_json JSON;

            ALTER TABLE opportunity_observations
            ADD COLUMN market_contexts_json JSON;
        """,
    ),
)
