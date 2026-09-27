# Monitoring operations

Existing native endpoints are dashboard `http://localhost:3000/d/drysua-training`,
Explore `/explore`, Prometheus `http://127.0.0.1:9090/` and exporter 127.0.0.1:9464.
Login is `admin` with the preserved private password, never admin/admin. Training
is OFF; monitoring changes must not launch a learner. Docker cutover remains
unqualified and daemon access denied; follow [Docker safety and admission](docker.md).
Historical native API/config/rule checks are not Docker or visual-browser evidence.

## Dashboard and memory settings

Reload with `?from=now-1h&to=now&refresh=30s` if an old tab retains refresh=5s.
Scrapes remain 5s. Time-series panels request 120 points at minimum 30s intervals
(inclusive endpoints may add one point); stats are instant queries. One CPU panel
shows selected logical CPUs rather than repeating panels. The 36-panel dashboard
retains progress, coverage, throughput, timing, PPO and whole-host/device resources.

Grafana-specific settings are GOMEMLIMIT=256MiB, GOGC=75, GODEBUG=disablethp=1;
the soft Go target is not an RSS/OOM guarantee. The native repair used 384 MiB high,
512 MiB hard, zero swap and 128 tasks, with no CPU quota or global THP change.
Its receipts and recovery data remain under
`artifacts/temp/monitoring-native-20260921/grafana-repair-20260921/`.
Do not replay that one-purpose launcher or edit a script while a long-lived shell
is reading it. The full repair diary remains at `git show 2ddb68b:docs/monitoring.md`.

## Data, identity and credentials

[Root Compose](../compose.yml) extends the [monitoring fragment](../monitoring/compose.yml)
and imports its secrets; it owns exporter/training lifecycle and overrides restart
to **no** and stop grace to 5s. The fragment alone uses on-failure:3 / 30s.
Use one root project consistently, not copied service blocks or conflicting exporters.
Configure private ignored `monitoring/local.env` (or root `docker/local.env`):

```dotenv
PROMETHEUS_DATA_DIRECTORY=/ABSOLUTE/PRIVATE/verified-copy/prometheus
GRAFANA_DATA_DIRECTORY=/ABSOLUTE/PRIVATE/verified-copy/grafana
MONITORING_UID=1000
MONITORING_GID=1000
GRAFANA_ADMIN_PASSWORD_FILE=/ABSOLUTE/PRIVATE/secrets/grafana_admin_password
GRAFANA_SECRET_KEY_FILE=/ABSOLUTE/PRIVATE/secrets/grafana_secret_key
PROMETHEUS_PORT=19090
GRAFANA_PORT=13000
```

Data paths must be existing, consistent copies—not live databases or second writers.
Bind mounts use create_host_path:false. Startup rejects missing/empty grafana.db
and missing Prometheus WAL/block history; these presence checks do not prove consistency.
UI port defaults are 9090/3000; 19090/13000 are staging ports, not current listeners.
Grafana's DRYSUA_PROMETHEUS_URL follows PROMETHEUS_PORT; the scrape target stays 9464.
Root configuration also requires the [training/exporter variables](docker.md).

Preserve the entire `artifacts/temp/monitoring-native-20260921` runtime, especially
data/prometheus, data/grafana, recovery copies, configuration, logs and provenance.
Retain its 0700 secrets directory and both 0600 files named above. Root Compose
requires explicit secret paths; the fragment's defaults point to these original files.
Grafana receives GF_SECURITY_ADMIN_PASSWORD__FILE and GF_SECURITY_SECRET_KEY__FILE.
Keep database, password and encryption key together; migrated accounts keep their
existing password. Never print, regenerate, broaden permissions or put secret values
in env files. File-backed Compose secrets retain host ownership; validate non-root
UID/user-namespace readability rather than assuming secret uid/mode remaps them.

## Future authorized migration

1. Inventory versions, edits, coverage and source paths; reserve private destinations
   and headroom. Preserve originals for rollback, including exporter journal/lock.
2. Obtain consistent backups: cleanly closed complete Grafana data including WAL/SHM,
   or a verified supported SQLite online backup; cleanly closed complete TSDB including
   WAL/head chunks, or an authorized supported snapshot including head data. Do not
   copy live databases arbitrarily or enable admin APIs/restart services for convenience.
3. Verify integrity, version/key compatibility and UID access on copies. Never reset
   originals to pass startup or merge independently written TSDB histories by overlay.
4. Stage only the UI at 19090/13000, scraping the existing native exporter. Compare
   history, login, edits, datasource results and persistence; account for sampling gaps.
5. Separately authorize final port/exporter handoff. Never overlap listeners at 9464,
   9090 or 3000, signal legacy processes without authority, or stop training for a UI
   test. Keep originals until rollback checks pass; unavailable handoff means defer.

## Commands (from repository root)

Configuration validation needs no daemon and should not dump real expanded env:

```sh
docker compose --env-file monitoring/local.env config --quiet
docker compose --env-file monitoring/local.env -f monitoring/compose.yml config --quiet
```

The following require approved daemon access, pinned images, copies and free ports;
they are not instructions to run during the current blocked deployment:

```sh
docker compose --env-file monitoring/local.env run --rm --no-deps \
  --entrypoint /bin/promtool prometheus check config /etc/prometheus/prometheus.yml
docker compose --env-file monitoring/local.env run --rm --no-deps \
  --workdir /etc/prometheus --entrypoint /bin/promtool prometheus test rules rules_test.yml
docker compose --env-file monitoring/local.env up -d --no-deps --pull never --no-build prometheus grafana
docker compose --env-file monitoring/local.env ps prometheus grafana
docker compose --env-file monitoring/local.env stats --no-stream prometheus grafana
docker compose --env-file monitoring/local.env logs --tail 100 prometheus grafana
docker compose --env-file monitoring/local.env stop grafana prometheus
```

UI-only stop preserves data/exporter/training; never use project-wide teardown or
volume removal for browser maintenance. Missing images fail (`pull_policy: never`).
For the existing native UI, forward to actual remote loopback listeners:

```sh
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:3000:127.0.0.1:3000 \
  -L 127.0.0.1:9090:127.0.0.1:9090 USER@TRAINING_HOST
```

For approved staging substitute 13000/19090 at both ends. Connection-refused means
the destination did not accept a connection, not a PromQL diagnosis. No LAN exposure:
exporter/Prometheus are unauthenticated, and Grafana uses local HTTP.

## Limits and graph interpretation

Exact digest pins live in monitoring/compose.yml (Prometheus 3.14.0, Grafana 13.2.2).
Both UI containers have 512 MiB hard RAM, no extra swap, 128 PIDs, read-only roots,
32 MiB tmp, dropped capabilities and no-new-privileges; Docker logs retain 10 MiB ×3.
Prometheus retains 14 days / 2 GB, scrapes/evaluates every 5s with 2s scrape timeout,
and bounds queries to concurrency 4, 10s and 500,000 samples. Retention is not a disk
quota: WAL/head/compaction and Grafana data need headroom. Keep plugin installation,
unused datasource backends, signup, anonymous access, embedding and public dashboards
disabled. Use personal Save as copies when edits must survive file provisioning.

[Metric semantics](metrics_schema.md) define coverage and availability. Rates precede
aggregation; zero denominators/missing observations mean No data, not invented zeros.
Training needs up + healthy/available state; resource availability is independent.
Writer-lock active is not progress; heartbeat is trainer-supplied, not scrape time.
ETA needs positive recent speed and heartbeat age [0,360) seconds; p95 is a bucket
estimate. Outcomes cover committed observations after U135, not reconstructed history.
