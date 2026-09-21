# Training graphs through Docker Compose

## Dashboard OOM mitigation — 2026-09-21

The live dashboard was backed up through the authenticated API, then updated
without restarting or signaling any process. Reload the existing browser tab at
**http://localhost:3000/d/drysua-training?from=now-1h&to=now&refresh=30s**; an already
open tab or `refresh=5s` URL can retain the old request rate until reloaded.

- Browser refresh is **30s**, rather than 5s. Prometheus scraping stays **5s**.
- Every time-series panel requests **120 points** with a **30s minimum interval**;
  inclusive range endpoints can return one extra point. Stat targets are explicitly
  instant-only. Metric expressions, coverage and availability gates are unchanged.
- A single full-width CPU chart returns all selected logical CPUs, including 16/32
  CPUs, instead of creating one repeated panel/query per core. The CPU variable
  refreshes on dashboard load, not on each time-range update.
- Dashboard UID and all 36 base panels remain; the saved live model differed from
  the source only in ID/version. The original API response is preserved at
  `artifacts/temp/monitoring-native-20260921/evidence/dashboard-oom-20260921-before.json`.

One bounded one-hour CPU request before the update took **0.000766s** directly in
Prometheus and **0.005229s** through Grafana. At that sample, Grafana's Go memory
limit was effectively unlimited, live heap allocation was **86,938,168 bytes**,
process RSS **434,999,296 bytes**, and cgroup memory **260,034,560 bytes**, with
**155,189,248 bytes** of anonymous THP. Memory had already fallen from the larger
values seen earlier. These snapshots and known OOM events do **not** establish
a heap leak or THP as the sole cause; short successful requests do not prove the
intermittent problem is eliminated.

Post-update readback confirmed the same Grafana PID **237668**, eight CPU series
with at most **121 points** each, and one-hour query times of **0.000891s** in
Prometheus / **0.005623s** through Grafana. Sampled cgroup memory was
**190,070,784 bytes**, process RSS **391,114,752 bytes**, and live Go heap
**82,841,760 bytes**. The current invocation's high-pressure count stayed at 4968
with no new OOM/PID events. Memory declined while GC also progressed; do not
attribute that decline solely to the dashboard change. Training advanced to
committed U253 during this check. Before/after summaries are preserved as
`evidence/grafana-oom-before.json` and `evidence/grafana-oom-after.json` in the
existing private monitoring runtime.

### Memory settings are now active, not restart-pending

An explicitly authorized **Grafana-only** repair replaced PID 237668 with PID
**253602** on port 3000. No service-manager control or privilege escalation was
used. The replacement self-joined this new delegated leaf before executing:

```text
/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/drysua-grafana-repair-20260921-live
```

Verified active settings: **`GOMEMLIMIT=256MiB`**, **`GOGC=75`**,
**`GODEBUG=disablethp=1`**. The metrics endpoint reports
`go_gc_gomemlimit_bytes=268435456` and `go_gc_gogc_percent=75`. Limits remain
**384 MiB high / 512 MiB max / zero swap / 128 tasks**. The process runs with no
new privileges, zero effective capabilities, host networking, private PID/proc and
32 MiB tmp, and a read-only root with only Grafana data/log directories writable.
There is no CPU quota, affinity, global thread setting or global THP change.

The replacement was first verified on port 13000 against a separate consistent
SQLite online backup. Only the identity-checked old Grafana child received
SIGTERM; its data was then reused after its process group became empty. Cutover to
readiness took **0.716 seconds**. Password, encryption key, dashboard edits and the
database were retained. The temporary stage instance was subsequently terminated
by its own verified Grafana PID, leaving port 13000 unused again.

The one-purpose launcher and private receipts are under
`artifacts/temp/monitoring-native-20260921/grafana-repair-20260921/`. The launcher
validates an empty, already-bounded cgroup before moving only its own new process;
it has no restart loop or general service-management behavior. This is a
**temporary native repair, not Docker migration**. It has no new boot/autorestart
promise. Only the exact legacy Grafana default-target enablement symlink was
preserved and removed after the replacement was healthy; the unit file and all
Prometheus/exporter/training links remain untouched.

One deployment correction matters: the earlier in-place edit of the shared shell
logger left old interpreters positioned inside changed file bytes. After the old
Grafana child exited, its wrapper produced a syntax error and the old manager
attempted a restart, which failed and hit its start limit. The shared logger was
restored to its original 865 bytes, and all existing readers were verified to be
at/beyond EOF. Memory settings now live in the **Grafana-specific launcher** and
Compose, not the shared logger. No Prometheus/exporter process was signaled or
restarted. Preserve `legacy-shutdown-journal.txt` and the repair receipts; do not
edit a shell script in place while a long-lived interpreter is reading it.

A Go soft memory target is not an RSS cap or a guarantee against OOM. Check the
repair's `normal-refresh-verification.json` for actual heap, RSS, anonymous THP,
event counters and query latency across ordinary 30-second refresh cycles.
Three cycles at 0/30/60 seconds passed: charged memory **141–143 MiB**, anonymous
THP **zero**, no memory high/OOM/PID events, and CPU+GPU query latencies
**9.706 / 4.613 / 5.426 ms**. Training advanced from U261 at preflight to U271;
rollback was not required. Login, dashboard content, frontend assets and SQLite
integrity were verified. This is bounded normal-use evidence, not a stress test.

Current host listeners are **3000 / 9090 / 9464**. Ports **13000 / 19090** below
are future Docker staging ports, not running services. For the existing deployment,
an SSH forward must target remote **127.0.0.1:3000**, not remote 13000. SSH's
`channel ... connection refused` means the forwarded destination did not accept
the connection; an OOM restart gap or a wrong destination can cause it. It does
not identify a PromQL failure. No visual-browser verification is claimed.

## Current status: Docker cutover is blocked, not completed

Docker Compose is the required operating interface for the complete stack,
including the trainer and stable binary metrics endpoint owned by the root
Compose configuration. See [Docker operations](docker.md) for that configuration.

**Docker daemon access is currently permission denied.** Compose v5.5.1 can parse
configuration without the daemon, but cannot deploy it. No permission escalation,
socket/group changes, image pulls, builds or container starts are part of this
preparation. Do not make the Docker socket world-writable. Membership in the
Docker group is effectively root-equivalent, not an innocuous workaround.
Rootless Docker is not automatically equivalent: validate host-loopback routing,
UID mapping, cgroup limits, GPU-toolkit/device access and whole-host resource
visibility before considering it a supported deployment.

**Except for the explicitly authorized Grafana-only repair above, existing
non-Docker monitoring and ongoing B40 training remain untouched.**
Do not signal or control those processes, remove their unit files/symlinks, alter
their data/credentials, or replay historical launch/guard commands. Historical
artifacts are evidence, not current operating instructions. An authorized and
technically possible monitoring handoff is still required; this document does
not claim that everything is already running in Docker.

Existing browser endpoints remain the legacy deployment until actual cutover:

- Dashboard: **http://localhost:3000/d/drysua-training**.
- Explore: **http://localhost:3000/explore**.
- Prometheus: **http://127.0.0.1:9090/**.
- Login: **`admin`**, using the existing private password file described below.

Training has real committed telemetry from coverage **after U135**, including
segment resumes. Earlier outcomes are not reconstructed. Current campaign
progress belongs in its `STATUS.json`, not a fixed number in these instructions.

## Root Compose inclusion contract

The root owner should include the fragment once:

```yaml
include:
  - monitoring/compose.yml
```

`monitoring/compose.yml` defines only services **`prometheus`** and **`grafana`**,
and secrets **`grafana_admin_password`** and **`grafana_secret_key`**. It declares
no project name, container names, named volumes or profiles. The root file owns
project naming, the stable exporter, training profiles and heavy-job isolation.
Monitoring stays light/default; training must remain an explicit root-level
operation. Do not duplicate the included service or secret names.

Use `include`, rather than copying service blocks or merging the fragment with
`-f`: include resolves its relative config/secret paths from `monitoring/`. If the
root owner instead uses `extends`, it must also import the two secrets and audit
every relative path; inheritance alone does not import top-level resources.

There is intentionally **no dependency on the root exporter service** in this
fragment. Explicit `--no-deps prometheus grafana` staging must not launch another
exporter onto an occupied port or affect the trainer. The root exporter contract:

- Expose the stable binary endpoint on **127.0.0.1:9464**, job **`drysua`**.
- Preserve the campaign metrics directory and writer-lock semantics described in
  [the metrics schema](metrics_schema.md). The exporter reads committed state;
  its lock probe needs access to `.metrics.writer.lock` without permission to
  rewrite committed state. The root owner supplies the correct mounts/identity.
- Supply host CPU/RAM/GPU telemetry with honest availability. Host networking alone
  does not grant GPU access or guarantee correct host resource visibility.
- Never run old and new exporters on port 9464 simultaneously. Exporter handoff
  is distinct from the browser transition and must not stop training.

## Configuration: explicit existing data and external secrets

Set these variables in a private, Git-ignored `monitoring/local.env` and pass it
explicitly to root Compose. The example contains paths only, never secret values:

```dotenv
# Absolute, separately prepared and verified copies; NEVER the live native paths.
PROMETHEUS_DATA_DIRECTORY=/ABSOLUTE/PRIVATE/migrated-monitoring/prometheus
GRAFANA_DATA_DIRECTORY=/ABSOLUTE/PRIVATE/migrated-monitoring/grafana
# Match the non-root owner of the copies and the 0600 credential files.
MONITORING_UID=1000
MONITORING_GID=1000
# Optional overrides; defaults point to the existing native credential files.
GRAFANA_ADMIN_PASSWORD_FILE=/ABSOLUTE/PRIVATE/secrets/grafana_admin_password
GRAFANA_SECRET_KEY_FILE=/ABSOLUTE/PRIVATE/secrets/grafana_secret_key
# Parallel UI readiness, while legacy 3000/9090/9464 remain occupied.
PROMETHEUS_PORT=19090
GRAFANA_PORT=13000
```

| Variable | Behavior |
| --- | --- |
| `PROMETHEUS_DATA_DIRECTORY` | Required absolute path to an existing, consistent TSDB copy or snapshot. |
| `GRAFANA_DATA_DIRECTORY` | Required absolute path to an existing, consistent Grafana data copy. |
| `MONITORING_UID`, `MONITORING_GID` | Default 1000:1000; choose a non-root identity that can read the secret files and write only the copied data. Account for user-namespace mapping. |
| `GRAFANA_ADMIN_PASSWORD_FILE` | Existing external password file; only its path enters Compose. |
| `GRAFANA_SECRET_KEY_FILE` | Existing encryption key; must be retained with the migrated database. |
| `PROMETHEUS_PORT` | Default 9090; changes only the loopback listener and Grafana's datasource URL. |
| `GRAFANA_PORT` | Default 3000; changes the loopback listener and root URL. |

Both data mounts use long-form bind syntax with **`create_host_path: false`**.
Missing variables fail configuration; missing directories fail container setup.
There are **no automatically created empty volumes**. Startup additionally rejects
Grafana data without a nonempty `grafana.db` and Prometheus data without a WAL
segment or block `meta.json`. These are fail-closed presence checks, **not** proof
that a copy is consistent. This fragment deliberately does not silently initialize
a new history when migrating an existing campaign.

The existing private runtime is:

```text
/home/alexstanovoy/Workspace/bots/drysua/artifacts/temp/monitoring-native-20260921
```

Preserve its `data/prometheus/`, `data/grafana/`, configuration, logs, units,
provenance and both **0600** files under its **0700** `secrets/` directory:

```text
secrets/grafana_admin_password
secrets/grafana_secret_key
```

The fragment defaults to these existing files using paths relative to itself.
Overrides should be absolute. It mounts them read-only; it does not copy,
regenerate, print or change permissions on them. Grafana's Docker entrypoint uses
`GF_SECURITY_ADMIN_PASSWORD__FILE` and `GF_SECURITY_SECRET_KEY__FILE`. The native
file-provider configuration is not a substitute for these Docker settings.

Local Compose file secrets are bind mounts, not encrypted storage. They retain
host-file ownership/permissions; do not assume secret `uid`/`mode` options remap a
0600 file. Validate the container UID and rootless mapping without broadening the
original permissions. Docker administrators can read such files. The admin
password initializes only a new database: a migrated database keeps its existing
account password. Preserve the encryption key even if current datasources have no
credentials. Never use `admin/admin` or put a real password in an env file.

## Safe history migration — future authorized work only

**Do none of this while Docker access and a permitted monitoring handoff remain
unavailable.** Training must continue throughout a UI transition; it writes its
own stable metrics directory, independent of Grafana and the Prometheus TSDB.

1. Inventory the original paths, versions, database/encryption key, existing
   dashboard edits and Prometheus coverage. Reserve new private destinations and
   disk headroom; keep original data/configuration immutable as the rollback source.
2. Obtain a **consistent** monitoring copy through a separately authorized
   maintenance procedure. For Grafana SQLite, copy its entire data directory only
   after Grafana has cleanly closed it, preserving any associated WAL/SHM files.
   A supported SQLite online-backup operation is an alternative only if the full
   deployed storage layout is understood and the resulting backup is verified.
   A raw copy of a live SQLite main file is not a backup.
3. For Prometheus, prefer a complete copy of a cleanly closed TSDB, including WAL
   and head chunks. A supported TSDB snapshot is an alternative **only if** its
   admin snapshot API is already available and its use is authorized; include head
   data and restore the snapshot's contents as the new data root. Do not enable an
   admin API or restart the current deployment merely to obtain a snapshot here.
   Never copy a live TSDB directory as if it were a consistent backup.
4. Verify copy integrity, version compatibility, UID/GID access and the matching
   Grafana encryption key. Adjust ownership **only on new copies** if needed.
   Do not mount a live source directory into a second database writer, even on
   alternate ports. Never rename/delete/reset the original databases or metrics
   journal to make startup pass.
5. Once access, images, valid copies and authorization exist, stage only the two
   UI containers at **19090/13000** using the commands below. They may scrape the
   existing stable native endpoint at 9464. This proves browser readiness, **not**
   complete Docker migration. Compare historical queries, dashboard edits, login,
   datasource health, resource availability and actual query results.
6. Plan the final monitoring-only handoff separately. Default 9090/3000 cannot
   be used until their old listeners are relinquished through an authorized
   mechanism; the same applies to exporter port 9464. No automatic legacy stop,
   disable, kill or cleanup is provided. If the permitted mechanism is unavailable,
   remain in the preserved/staged state instead of interrupting training.

Prometheus processes cannot merge arbitrary independently written TSDB histories
by copying files over one another. Keep a single validated successor data path,
account for any sampling gap, and retain the source backup until comparison and
rollback checks pass. Do not advertise history before first deployment or silently
replace the campaign with an empty database.

## Docker Compose commands

From the repository root, this is safe now and does not need daemon access:

```sh
docker compose --env-file monitoring/local.env config --quiet
```

The root configuration may require additional variables documented in
`docs/docker.md`. Until that file is present, the fragment alone can be checked
with `docker compose --env-file monitoring/local.env -f monitoring/compose.yml config --quiet`.
Do not print full environments or resolved configuration into shared logs.

**The following commands are for later, authorized Docker operation only**, with
verified copies, free selected ports and the pinned images available. No commands
in this preparation stop or change any live service. Image acquisition and heavier
validation belong to the root owner's Docker workflow, not the historical guard.

```sh
# Validate in short-lived containers without starting dependencies.
docker compose --env-file monitoring/local.env run --rm --no-deps \
    --entrypoint /bin/promtool prometheus check config /etc/prometheus/prometheus.yml
docker compose --env-file monitoring/local.env run --rm --no-deps \
    --workdir /etc/prometheus --entrypoint /bin/promtool prometheus \
    test rules rules_test.yml

# Explicit UI-only startup; never implicitly start a conflicting exporter/trainer.
docker compose --env-file monitoring/local.env up -d --no-deps --pull never --no-build prometheus grafana
docker compose --env-file monitoring/local.env ps prometheus grafana
docker compose --env-file monitoring/local.env stats --no-stream prometheus grafana
docker compose --env-file monitoring/local.env logs --tail 100 prometheus grafana

# Docker UI-only stop preserves data and does not stop the exporter or training.
docker compose --env-file monitoring/local.env stop grafana prometheus
```

Do not use project-wide teardown or volume-removal commands for a browser change.
Use the same root project and env file consistently. The fragment has
`pull_policy: never`; missing images are an explicit blocker, not an implicit
download. Root Compose owns exporter/training lifecycle commands separately.

## Image pins, limits and access

GitHub release metadata and Docker Hub manifest hashes were verified on 2026-09-21:

| Image | Published UTC | Pinned manifest-list SHA256 |
| --- | --- | --- |
| `prom/prometheus:v3.14.0` | 2026-08-18 08:49:40 | `5ce7540c3c00ef4ab0c9d2c995c6a5b9c421f44b4a115d97a2c7af3b1c21cbb0` |
| `grafana/grafana:13.2.2` | 2026-09-15 12:21:32 | `ac461fb352abc50da10a51c7d02462e9c05488f11f53f14b3ad79a8145f638a0` |

Sources: [Prometheus release API](https://api.github.com/repos/prometheus/prometheus/releases/tags/v3.14.0),
[Grafana release API](https://api.github.com/repos/grafana/grafana/releases/tags/v13.2.2),
and the corresponding authenticated manifest endpoints at
`https://registry-1.docker.io/v2/prom/prometheus/manifests/v3.14.0` and
`https://registry-1.docker.io/v2/grafana/grafana/manifests/13.2.2`.
Grafana 13.2.2 is covered by the **already allowed security-patch exception** to
the one-week age rule; no additional exception or waiting period is needed.
Manifest hashes establish identity, not publisher-signature or Docker runtime
verification. Historical native tool hashes remain in the preserved provenance.

- Linux host networking; application listeners bind **127.0.0.1**, never wildcard
  addresses. A remote daemon or incompatible rootless namespace is not equivalent.
- Each UI container: **512 MiB RAM**, no additional swap, **128 PIDs**, read-only
  root filesystem, dropped capabilities, no-new-privileges and a 32 MiB `/tmp`.
- Logs: Docker `json-file`, **10 MiB × 3** per container. Restart: **`on-failure:3`**;
  diagnose failures rather than raising limits or adding a retry loop.
- Prometheus retention: **14 days / 2 GB**; scrape/evaluation **5s**, timeout **2s**;
  bounded scrape body/samples/labels and query concurrency/time/sample count.
  Retention is not a filesystem quota: WAL/head/compaction need headroom and
  Grafana's database is not automatically disk-quota bounded.
- Only the required Prometheus bundled datasource backend is enabled. Historical
  native testing showed unused backends exhausted 128 tasks; they are disabled
  rather than increasing the limit. Standard visualization panels remain usable.
- Anonymous access, signup, embedding, public dashboards and plugin installation
  are disabled. Preserve these settings while migrating. Login as `admin`; edit
  panels and save in the browser. Personal **Save as** copies avoid later file
  provisioning updates overwriting edits. The datasource itself is provisioned.

For a later staged Docker UI, use **http://localhost:13000/d/drysua-training** and
**http://127.0.0.1:19090/**. Grafana's datasource follows `PROMETHEUS_PORT` through
`DRYSUA_PROMETHEUS_URL`; Prometheus's exporter target remains 127.0.0.1:9464.
For a remote browser, explicitly forward only loopback ports:

```sh
ssh -N -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:13000:127.0.0.1:13000 \
    -L 127.0.0.1:19090:127.0.0.1:19090 USER@TRAINING_HOST
```

Use 3000/9090 instead after final default-port cutover. Direct LAN exposure is not
configured: Prometheus/exporter have no authentication and Grafana uses local HTTP.

## Metric interpretation and verification status

The authoritative names and persistence semantics are in
[metrics_schema.md](metrics_schema.md). No dashboard queries changed for this
Compose integration. The 36-panel dashboard includes configured `parallel_worlds`
and `games_per_update`, outcome coverage, throughput, timing mean/p95, PPO gauges,
progress/ETA, one dynamically populated logical-CPU chart, GPU/VRAM/temperature
and host RAM.

- Win rates include win/loss/draw/time-cap **once each**. Rates are calculated
  before aggregation; zero denominators and missing observations yield **No data**,
  not invented zeroes or NaN. Cumulative outcomes cover committed observations
  **after** `metrics_start_update`, not all earlier checkpoint history.
- Training values require endpoint up, `metrics_available=1` and
  `metrics_state_healthy=1`. CPU/RAM/GPU have independent collection availability
  gates and describe the **whole host/device**, not individual bots.
- Stable-exporter `active` means writer lock held, not learner progress. Heartbeat
  timestamps advance at initialization/checkpoint/target refresh, not each scrape.
  ETA requires positive recent speed and heartbeat age in **[0, 360) seconds**;
  a legitimate 250-second update is not falsely stale. Timing histograms persist
  across segments and describe committed updates; p95 is a bucket estimate.
- Episode audit/control and checkpoint/final protocol records remain logs;
  periodic metric summaries are replaced by Prometheus. No historical outcome
  reconstruction, trainer change or checkpoint mutation belongs to this UI move.

Historical native verification passed config/ten rules, all **13 declarative rule
cases**, real exposition lint, **44 dashboard queries**, authentication, save/query
APIs, resource availability and restart history. That is **not Docker migration
evidence**. This reconciliation performs only YAML/JSON/shell-syntax and Compose
model checks; it does not execute promtool, profile/rule tests, the old guard,
service controls or any live migration. Docker startup, mount/UID compatibility,
copied-history integrity and browser readiness on the new stack remain unverified.
No visual browser automation is claimed.

Static checks passed for root inclusion/project/path resolution, required-data
rejection, non-creating bind mounts, external secret/UID overrides, default and
staging port wiring, YAML/JSON parsing and startup shell syntax. No data directories
or secret files were created, copied or read by these checks.
