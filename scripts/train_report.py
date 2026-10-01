"""Read-only training statistics and a self-contained HTML dashboard from trainer logs.

Works on a campaign directory (sessions or the old per-invocation layout), a
plain directory of logs, or one log file (optionally gzip-compressed). Lines
are parsed generically, so new numeric fields show up without code changes:

* `episode: key=value ...` - one terminal episode;
* `level=INFO event=<name> key=value ...` - structured events;
* `progress: update N, key value, ...` - the statistics of update N, logged
  every update; they are exposed as `checkpoint.<field>` metrics;
* `checkpoint: update N` - the durable commit of update N; the trainer commits
  only periodically, so it trails the progress lines.

Episodes and events accumulate until the next progress line. A process restart
(`### ` separator, file boundary, or `annealed: updates=` header) discards the
unattributed tail and every update after the last durable checkpoint: the
trainer recomputes those and logs them again.
"""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import time

import train_io as io

REPORT_SCHEMA = "drysua-training-report/v2"
MAX_LINE = 64 * 1024
MAX_FILE_BYTES = 4 * 1024 ** 3
MAX_FILES = 100000
MAX_UPDATES = 100000
MAX_SERIES = 2000
OUTCOMES = ("Win", "Draw", "Loss")
ACTION_KINDS = ("Continue", "Stop", "MovePoint", "FollowUnit", "Hold", "AttackMovePoint", "AttackUnit", "Cast",
                "Use", "PutPoint", "PutUnit", "Take", "Buy", "Sell", "Swap", "Learn")
TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)=(\[[^\]]*\]|\S+)")
EVENT = re.compile(r"level=([A-Z]+) event=([A-Za-z0-9_]+)\s*(.*)")
PREFIX = re.compile(r"([a-z][a-z ]{0,40}[a-z]):\s+(.*)")
# Identity or cumulative fields are no per-update measurement.
IGNORED_FIELDS = {"stream", "map", "update", "update_index", "completed_updates", "updates", "dropped_logs",
                  "start_game", "previous_generation", "start_update", "optimizer_step", "samples_total"}
IGNORED_EVENTS = {"map2_training_reward"}
CATEGORY_EXCLUDED = {"outcome", "opponent", "actions"}
# League snapshots of the learner (`u0040`) come and go; they share one series.
LEAGUE_LABEL = re.compile(r"u[0-9]{4,}")
MAX_CATEGORIES = 8


def number(text):
    if text in ("true", "false"):
        return 1.0 if text == "true" else 0.0
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def key_value_fields(text):
    fields = {}
    for name, raw in TOKEN.findall(text):
        if raw.startswith("["):
            values = [number(part.strip()) for part in raw[1:-1].split(",") if part.strip()]
            fields[name] = values if all(value is not None for value in values) else raw
        else:
            value = number(raw)
            fields[name] = raw if value is None else value
    return fields


def prose_fields(text):
    """`progress: update 3, policy loss -0.1, KL stop false` -> {update: 3, policy_loss: -0.1, ...}."""
    fields = {}
    for part in text.split(", "):
        words = part.split()
        if len(words) >= 2 and (value := number(words[-1])) is not None:
            fields["_".join(words[:-1]).lower()] = value
    return fields


def classify(line):
    """(kind, name, fields) of one log line, or None for lines without statistics."""
    if line.startswith("### "):
        return ("restart", None, {})
    match = EVENT.match(line)
    if match:
        return ("event", match.group(2), key_value_fields(match.group(3)))
    match = PREFIX.match(line)
    if not match:
        return None
    name, rest = match.group(1).replace(" ", "_"), match.group(2)
    if name == "episode":
        return ("episode", name, key_value_fields(rest))
    if name in ("progress", "checkpoint"):
        return (name, name, prose_fields(rest))
    if name == "annealed":
        fields = key_value_fields(rest)
        return ("restart", name, fields) if "games" in fields and "parallel" in fields else ("event", name, fields)
    return None


def explicit_update(fields):
    if isinstance(fields.get("update_index"), float):
        return int(fields["update_index"]) + 1
    for name in ("completed_updates", "update"):
        if isinstance(fields.get(name), float):
            return int(fields[name])
    return None


class Pending:
    """Episodes and events of the update in flight."""

    def __init__(self):
        self.episodes, self.events = [], defaultdict(list)


class LogModel:
    """Per-update aggregates, keeping only updates the trainer will not recompute."""

    def __init__(self):
        self.updates, self.pending = {}, Pending()
        self.process_starts, self.last_progress, self.durable = [], 0, 0

    def feed(self, line):
        parsed = classify(line)
        if parsed is None:
            return
        kind, name, fields = parsed
        if kind == "restart":
            self.restart()
            return
        if kind == "episode":
            self.pending.episodes.append(fields)
        elif kind == "progress":
            self.record_progress(fields)
        elif kind == "checkpoint":
            self.durable = max(self.durable, int(fields.get("update", 0)))
        elif name not in IGNORED_EVENTS:
            update = explicit_update(fields)
            if update is not None and update <= self.last_progress and update in self.updates:
                merge(self.updates[update]["metrics"], event_metrics({name: [fields]}), overwrite=False)
            else:
                self.pending.events[name].append(fields)

    def restart(self):
        """A new trainer process resumes from the last durable checkpoint and replays everything after it."""
        self.pending = Pending()
        self.updates = {update: record for update, record in self.updates.items() if update <= self.durable}
        self.last_progress = self.durable
        start = self.durable + 1
        if not self.process_starts or self.process_starts[-1] != start:
            self.process_starts.append(start)

    def record_progress(self, fields):
        update = fields.get("update")
        if update is None or not 1 <= update <= MAX_UPDATES:
            return
        update = int(update)
        metrics = event_metrics(self.pending.events)
        merge(metrics, {f"checkpoint.{key}": value for key, value in fields.items()
                        if key not in IGNORED_FIELDS and key != "samples" and not key.startswith("session_")})
        record = episode_record(self.pending.episodes)
        record.update(update=update, process=len(self.process_starts), metrics=merge(record.pop("metrics"), metrics))
        self.updates[update] = record
        self.last_progress = update
        self.pending = Pending()
        if len(self.updates) > MAX_UPDATES:
            raise ValueError(f"log exceeds {MAX_UPDATES} updates")

    def records(self):
        return [self.updates[update] for update in sorted(self.updates)]


def merge(target, source, overwrite=True):
    for key, value in source.items():
        if overwrite or key not in target:
            target[key] = value
    if len(target) > MAX_SERIES:
        raise ValueError(f"log exceeds {MAX_SERIES} series per update")
    return target


def mean_fields(lines, prefix):
    sums, counts = defaultdict(float), defaultdict(int)
    for fields in lines:
        scope = fields.get("scope")
        name = f"{prefix}.{scope}" if isinstance(scope, str) else prefix
        for key, value in fields.items():
            if isinstance(value, float) and key not in IGNORED_FIELDS:
                if key.endswith("_ns"):
                    key, value = key[:-3] + "_s", value / 1e9
                sums[f"{name}.{key}"] += value
                counts[f"{name}.{key}"] += 1
    return {key: sums[key] / counts[key] for key in sums}


def opponent_group(label):
    return "league" if LEAGUE_LABEL.fullmatch(label) else label


def pool_metrics(lines):
    """Per-opponent PFSP window win rate and sampling probability of the last published mixture."""
    games, score, probability = defaultdict(float), defaultdict(float), defaultdict(float)
    last = max((fields.get("update", 0.0) for fields in lines), default=0.0)
    for fields in lines:
        if fields.get("update", 0.0) != last or not isinstance(fields.get("scope"), str):
            continue
        group = opponent_group(fields["scope"])
        games[group] += fields.get("games", 0.0)
        score[group] += fields.get("score", 0.0)
        probability[group] += fields.get("probability", 0.0)
    metrics = {f"opponent_pool.{group}.probability": value for group, value in probability.items()}
    metrics.update({f"opponent_pool.{group}.win_rate": score[group] / count
                    for group, count in games.items() if count})
    return metrics


def event_metrics(events):
    metrics = {}
    for name, lines in events.items():
        metrics.update(pool_metrics(lines) if name == "opponent_pool" else mean_fields(lines, name))
    return metrics


def episode_record(episodes):
    outcomes = {outcome: 0 for outcome in OUTCOMES}
    by_opponent = defaultdict(lambda: {outcome: 0 for outcome in OUTCOMES})
    actions = [0.0] * len(ACTION_KINDS)
    categories = defaultdict(lambda: defaultdict(int))
    for fields in episodes:
        outcome = fields.get("outcome")
        if outcome not in outcomes:
            continue
        outcomes[outcome] += 1
        by_opponent[opponent_group(str(fields.get("opponent", "all")))][outcome] += 1
        for key, value in fields.items():
            if isinstance(value, str) and key not in CATEGORY_EXCLUDED:
                categories[key][value] += 1
        counts = fields.get("actions")
        if isinstance(counts, list) and len(counts) == len(actions):
            actions = [total + count for total, count in zip(actions, counts)]
    metrics = mean_fields(episodes, "episode")
    ticks = [fields["tick"] for fields in episodes if isinstance(fields.get("tick"), float)]
    if ticks:
        metrics.update({"episode.tick_min": min(ticks), "episode.tick_max": max(ticks)})
    return {"games": sum(outcomes.values()), "outcomes": outcomes, "by_opponent": dict(by_opponent),
            "actions": actions, "metrics": metrics,
            "categories": {key: dict(counts) for key, counts in categories.items()}}


def log_sources(target):
    """Ordered log files of a campaign, a log directory, or one file."""
    target = io.private_path(target)
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise ValueError(f"no such log file or directory: {target}")
    for layout in ("sessions", "invocations"):
        if (target / layout).is_dir():
            files = sorted(path / "payload.log" for path in (target / layout).iterdir()
                           if (path / "payload.log").is_file())
            return bounded_files(files)
    preferred = [target / name for name in ("payload.log", "payload.log.gz") if (target / name).is_file()]
    return bounded_files(preferred or sorted(path for path in target.iterdir()
                                             if path.is_file() and path.name.endswith((".log", ".log.gz"))))


def bounded_files(files):
    if len(files) > MAX_FILES:
        raise ValueError(f"more than {MAX_FILES} log files")
    return files


def read_logs(files):
    model = LogModel()
    for path in files:
        model.restart()
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(io.private_path(path), "rb") as source:
            consumed = 0
            for raw in source:
                consumed += len(raw)
                if consumed > MAX_FILE_BYTES:
                    raise ValueError(f"log exceeds {MAX_FILE_BYTES} bytes: {path}")
                if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
                    continue  # An overlong or still-being-written line carries no statistics.
                model.feed(raw.decode("utf-8", "replace").rstrip("\n"))
    return model


def tally(records):
    wins = sum(record["outcomes"]["Win"] for record in records)
    draws = sum(record["outcomes"]["Draw"] for record in records)
    losses = sum(record["outcomes"]["Loss"] for record in records)
    games = wins + draws + losses
    return {"games": games, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": round(wins / games, 4) if games else None}


def wilson(wins, games, z=1.96):
    """95% Wilson score interval of a win rate; robust for few games and rates near 0 or 1."""
    if games == 0:
        return None, None
    rate = wins / games
    center = (rate + z * z / (2 * games)) / (1 + z * z / games)
    margin = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / (1 + z * z / games)
    return max(0.0, center - margin), min(1.0, center + margin)


def campaign_info(target):
    """Best-effort identity and progress of a campaign directory; logs remain the source of statistics."""
    info = {}
    if not target.is_dir():
        return info
    for name in ("manifest.json", "status.json"):
        try:
            value = io.read_json(target / name) if (target / name).is_file() else {}
        except (OSError, ValueError):
            continue
        config = value.get("config") if isinstance(value.get("config"), dict) else {}
        for key, found in (("campaign_id", value.get("campaign_id")), ("phase", value.get("phase")),
                           ("total_updates", config.get("total_updates"))):
            if found is not None:
                info[key] = found
    return info


def build_report(target, block=10):
    """Read-only statistics of the logged updates in the logs under `target`."""
    target = io.private_path(target)
    io.bounded_integer(block, "report block", 1, MAX_UPDATES)
    model = read_logs(log_sources(target))
    records = model.records()
    info = campaign_info(target)
    total = info.get("total_updates")
    last = records[-1]["update"] if records else 0
    timed = [record["metrics"]["training_update_timing.elapsed_s"] for record in records[-20:]
             if "training_update_timing.elapsed_s" in record["metrics"]]
    seconds = round(sum(timed) / len(timed), 3) if timed else None
    remaining = max(0, total - last) if isinstance(total, int) else None
    opponents = sorted({name for record in records for name in record["by_opponent"]})
    transitions = [record["update"] for record in records
                   if "adaptive_environment_transition.generation" in record["metrics"]]
    return {
        "schema": REPORT_SCHEMA, "source": str(target), "campaign_id": info.get("campaign_id"),
        "phase": info.get("phase"), "total_updates": total, "updates": len(records), "last_update": last,
        "processes": len(model.process_starts), "in_flight_games": len(model.pending.episodes),
        "games": sum(record["games"] for record in records), "overall": tally(records),
        "by_opponent": {name: tally([{"outcomes": record["by_opponent"].get(name, dict.fromkeys(OUTCOMES, 0))}
                                     for record in records]) for name in opponents},
        "blocks": [dict(tally(records[start:start + block]), first_update=records[start]["update"],
                        last_update=records[min(start + block, len(records)) - 1]["update"])
                   for start in range(0, len(records), block)],
        "recent": {f"last{count}": tally(records[-count:]) if len(records) >= count else None for count in (10, 20)},
        "per_update": [dict(update=record["update"], wins=record["outcomes"]["Win"],
                            losses=record["outcomes"]["Loss"], draws=record["outcomes"]["Draw"])
                       for record in records],
        "environments": {"transitions": transitions, "generation": last_generation(records)},
        "timing": {"seconds_per_update": seconds,
                   "eta_seconds": None if seconds is None or remaining is None else round(remaining * seconds)},
    }, model


def last_generation(records):
    for record in reversed(records):
        for key in ("annealed.generation", "adaptive_environment_transition.generation"):
            if key in record["metrics"]:
                return int(record["metrics"][key])
    return None


def record_text(record):
    rate = record["win_rate"]
    return f"{record['wins']}-{record['losses']}-{record['draws']} win={'n/a' if rate is None else f'{rate:.1%}'}"


def format_report(value):
    total = value["total_updates"] if value["total_updates"] is not None else "?"
    lines = [f"source {value['source']}",
             f"campaign {value['campaign_id'] or 'n/a'} phase={value['phase'] or 'n/a'} "
             f"updates={value['last_update']}/{total} games={value['games']} processes={value['processes']}",
             f"overall {record_text(value['overall'])}"]
    if len(value["by_opponent"]) > 1:
        lines += [f"opponent {name} {record_text(record)}" for name, record in value["by_opponent"].items()]
    lines += [f"block {block['first_update']:04d}-{block['last_update']:04d} {record_text(block)}"
              for block in value["blocks"]]
    lines += [f"{name} {record_text(record)}" for name, record in value["recent"].items() if record is not None]
    if value["per_update"]:
        lines.append("per-update wins: " + ",".join(str(entry["wins"]) for entry in value["per_update"]))
    environments = value["environments"]
    if environments["transitions"] or environments["generation"] is not None:
        lines.append(f"environments transitions={len(environments['transitions'])} "
                     f"generation={environments['generation']}")
    timing = value["timing"]
    seconds, eta = timing["seconds_per_update"], timing["eta_seconds"]
    lines.append("timing n/a" if seconds is None else
                 f"timing {seconds}s/update eta={'n/a' if eta is None else f'{eta // 3600}h{eta % 3600 // 60:02d}m'}")
    return "\n".join(lines)


MAX_BARS = 400
PPO_PATTERN = re.compile(r"^checkpoint\.(policy_loss|value_loss|entropy|kl|kl_stop)$|clip_frac|explained_var|grad_norm")
TIMING_KEYS = ("training_update_timing.elapsed_s", "training_update_timing.collection_s",
               "training_update_timing.optimization_s")
ENVIRONMENT_KEYS = ("annealed.generation", "annealed.scale_bp", "adaptive_environment_transition.generation")


def rounded(value):
    return None if value is None else float(f"{value:.6g}")


def series(name, values, **extra):
    return dict(name=name, values=[rounded(value) for value in values], **extra)


def chart(title, kind, entries, unit="", x=None, domain=None, note=None):
    return {"title": title, "kind": kind, "series": entries, "unit": unit, "x": x, "domain": domain, "note": note}


def binned(records):
    """Consecutive update bins so a bar chart never exceeds MAX_BARS bars."""
    size = max(1, math.ceil(len(records) / MAX_BARS))
    return [records[start:start + size] for start in range(0, len(records), size)]


def outcome_charts(records, window):
    bins = binned(records)
    x = [chunk[-1]["update"] for chunk in bins]
    size = len(bins[0]) if bins else 1
    bars = chart("Games per update" if size == 1 else f"Games per {size} updates", "bars",
                 [series(outcome, [sum(record["outcomes"][outcome] for record in chunk) for chunk in bins])
                  for outcome in OUTCOMES], unit="games", x=x)
    opponents = sorted({name for record in records for name in record["by_opponent"]})
    groups = [("all", lambda record: record["outcomes"])]
    if len(opponents) > 1:
        groups += [(name, lambda record, name=name: record["by_opponent"].get(name, dict.fromkeys(OUTCOMES, 0)))
                   for name in opponents]
    lines = []
    for name, outcomes in groups[:8]:
        rates, lows, highs = [], [], []
        for index in range(len(records)):
            window_records = records[max(0, index - window + 1):index + 1]
            wins = sum(outcomes(record)["Win"] for record in window_records)
            games = sum(sum(outcomes(record).values()) for record in window_records)
            low, high = wilson(wins, games)
            rates.append(100 * wins / games if games else None)
            lows.append(None if low is None else 100 * low)
            highs.append(None if high is None else 100 * high)
        lines.append(series(name, rates, low=[rounded(value) for value in lows],
                            high=[rounded(value) for value in highs]))
    rate = chart(f"Win rate, rolling {window} updates", "lines", lines, unit="%", domain=[0, 100],
                 note="Band: Wilson 95% interval over the window's games.")
    return [bars, rate]


def action_chart(records):
    bins = binned(records)
    totals = [sum(record["actions"][kind] for record in records) for kind in range(len(ACTION_KINDS))]
    if not any(totals):
        return []
    top = sorted(range(len(ACTION_KINDS)), key=lambda kind: -totals[kind])[:7]
    entries = []
    for kind in top + [None]:
        values = []
        for chunk in bins:
            actions = [sum(record["actions"][index] for record in chunk) for index in range(len(ACTION_KINDS))]
            total = sum(actions)
            share = (actions[kind] if kind is not None else sum(actions[index] for index in range(len(actions))
                                                                  if index not in top))
            values.append(100 * share / total if total else None)
        entries.append(series(ACTION_KINDS[kind] if kind is not None else "Other", values))
    return [chart("Action kind share", "bars", entries, unit="%", x=[chunk[-1]["update"] for chunk in bins],
                  domain=[0, 100])]


def held(values):
    """Forward-fills a state that is only logged when it changes."""
    result, last = [], None
    for value in values:
        last = value if value is not None else last
        result.append(last)
    return result


def category_charts(records):
    """Share of each value of a low-cardinality string episode field, e.g. `end_reason`."""
    names = sorted({name for record in records for name in record["categories"]})
    charts, bins = [], binned(records)
    for name in names:
        values = sorted({value for record in records for value in record["categories"].get(name, {})})
        if len(values) > MAX_CATEGORIES:
            continue
        entries = []
        for value in values:
            shares = []
            for chunk in bins:
                counts = [record["categories"].get(name, {}) for record in chunk]
                total = sum(sum(count.values()) for count in counts)
                shares.append(100 * sum(count.get(value, 0) for count in counts) / total if total else None)
            entries.append(series(value, shares))
        charts.append(chart(f"episode {name} share", "bars", entries, unit="%",
                            x=[chunk[-1]["update"] for chunk in bins], domain=[0, 100]))
    return charts


def metric(records, key):
    return [record["metrics"].get(key) for record in records]


def pool_charts(records, keys, used):
    """PFSP window win rate and sampling probability per opponent, as the trainer published them."""
    charts = []
    for field, title, scale in (("win_rate", "Opponent win rate (last 100 games each)", 100),
                                ("probability", "Opponent sampling probability", 100)):
        selected = [key for key in keys if key.startswith("opponent_pool.") and key.endswith("." + field)]
        used.update(selected)
        if selected:
            entries = [series(key.split(".")[1], [None if value is None else scale * value
                                                  for value in metric(records, key)]) for key in selected]
            charts.append(chart(title, "lines", entries, unit="%", domain=[0, 100]))
    return charts


def potential_charts(records, keys, used):
    """Held-out win-ranking AUC of the learned and the hand potential, by game minute."""
    charts = []
    for kind in ("learned", "hand"):
        pattern = re.compile(rf"win_model\.auc_{kind}_m(\d+)")
        selected = sorted((key for key in keys if pattern.fullmatch(key)),
                          key=lambda key: int(pattern.fullmatch(key).group(1)))
        used.update(selected)
        if selected:
            charts.append(chart(f"{kind} potential AUC by game minute", "lines",
                                [series(f"min {pattern.fullmatch(key).group(1)}", held(metric(records, key)))
                                 for key in selected], domain=[0.3, 1.0],
                                note="On the games finished since the previous refit; 0.5 is chance."))
    return charts


def dashboard_sections(records, window):
    """Chart sections; every numeric series appears exactly once, known ones in curated charts."""
    keys = sorted({key for record in records for key in record["metrics"]})[:MAX_SERIES]
    used = set()

    def singles(selected):
        used.update(selected)
        return [chart(key, "lines", [series(key, metric(records, key))]) for key in selected]

    timing = [key for key in TIMING_KEYS if key in keys]
    used.update(timing)
    rate = [None if not samples or not seconds else samples / seconds for samples, seconds in
            zip(metric(records, "training_update_timing.samples"), metric(records, "training_update_timing.elapsed_s"))]
    sections = [("Outcomes", False, outcome_charts(records, window) + action_chart(records)),
                ("Timing", True, [chart("Update phases", "lines", [series(key.split(".")[-1], metric(records, key))
                                                                     for key in timing], unit="s"),
                                  chart("Retained samples per second", "lines", [series("samples/s", rate)])])]
    if "episode.tick" in keys:
        used.update({"episode.tick", "episode.tick_min", "episode.tick_max"})
        length = series("mean", metric(records, "episode.tick"),
                        low=[rounded(value) for value in metric(records, "episode.tick_min")],
                        high=[rounded(value) for value in metric(records, "episode.tick_max")])
        others = [key for key in keys if key.startswith("episode.") and key not in used]
        sections.append(("Episodes", True, [chart("Episode length (ticks)", "lines", [length],
                                                  note="Band: shortest to longest.")]
                         + category_charts(records) + singles(others)))
    sections.append(("Learned potential", True, potential_charts(records, keys, used)))
    sections.append(("PPO", True, singles([key for key in keys if PPO_PATTERN.search(key)])))
    sections.append(("Opponents", True, pool_charts(records, keys, used)))
    environment = [key for key in ENVIRONMENT_KEYS if key in keys]
    used.update(environment)
    sections.append(("Environment", True, [chart(key, "lines", [series(key, held(metric(records, key)))],
                                                 note="Held between the updates that report it.")
                                           for key in environment]))
    sections.append(("Reward components (mean per episode)", True,
                     singles([key for key in keys if re.match(r"map2_episode_reward\.reward_", key)])))
    sections.append(("Other metrics", True, singles([key for key in keys if key not in used])))
    return [{"title": title, "grid": grid, "charts": charts} for title, grid, charts in sections if charts]


def evaluation_sections(target):
    """Frozen pool evaluation of a campaign's `eval/` store, by snapshot update."""
    store = Path(target) / "eval"
    if not store.is_dir():
        return []
    import eval_pool  # not a frozen training source: only reports and `eval` read the store
    value = eval_pool.report(eval_pool.Store(store))
    snapshots, averages = {}, {}
    for candidate in value["candidates"].values():
        match = re.fullmatch(r"u(\d+)(?:-avg(\d+))?", candidate["name"])
        if match:
            rows = averages.setdefault(f"avg{match.group(2)}", {}) if match.group(2) else snapshots
            rows[int(match.group(1))] = candidate
    x = sorted(snapshots.keys() | {update for rows in averages.values() for update in rows})
    if not x:
        return []
    charts = evaluation_charts(eval_pool, value, x, snapshots, averages)
    return [{"title": f"Frozen pool evaluation ({value['context'][:16]}...)", "grid": True, "charts": charts}]


def aligned(x, rows, pick):
    """`pick(row)` at every snapshot update, `None` where the candidate was not rated."""
    return [pick(rows[update]) if update in rows else None for update in x]


def rate_series(name, records):
    """Score percentages with Wilson bands of `record`s (`None` or empty where not played)."""
    played = [record if record and record["games"] else None for record in records]
    return series(name, [None if record is None else 100 * record["win_rate"] for record in played],
                  low=[None if record is None else rounded(100 * record["ci95"][0]) for record in played],
                  high=[None if record is None else rounded(100 * record["ci95"][1]) for record in played])


def elo_series(name, ratings):
    return series(name, [None if rating is None else rating["elo"] for rating in ratings],
                  low=[None if rating is None else rounded(rating["elo"] - 1.96 * rating["se"]) for rating in ratings],
                  high=[None if rating is None else rounded(rating["elo"] + 1.96 * rating["se"]) for rating in ratings])


def side_rate(record, side):
    games = record["all"]["sides"][side]["games"]
    return 100 * record["all"]["sides"][side]["wins"] / games if games else None


def evaluation_charts(eval_pool, value, x, snapshots, averages):
    names = value["names"]
    opponents = sorted({key for row in snapshots.values() for key in row["by_opponent"]},
                       key=lambda key: names.get(key, key))
    anchor = next((names.get(key, key) for key, rating in value["ratings"].items() if rating["anchor"]), "n/a")

    def at(pick):
        return aligned(x, snapshots, pick)

    charts = [
        chart(f"Pool Elo (anchor {anchor} = 0)", "lines",
              [elo_series("snapshot", at(lambda row: row["rating"]))]
              + [elo_series(name, aligned(x, rows, lambda row: row["rating"])) for name, rows in sorted(averages.items())],
              unit="Elo", x=x, note="Bradley-Terry over every stored game of this context; band: 95% interval."),
        chart("Robustness: worst case and held-out", "lines",
              [rate_series("worst case", at(lambda row: row["worst"])),
               rate_series("held-out", at(lambda row: row["held_out"])),
               rate_series("train", at(lambda row: row["train"]))], unit="%", x=x, domain=[0, 100],
              note="Score (draws half). Worst case: the lowest per-opponent score. Band: Wilson 95%."),
        chart("Score per opponent", "lines",
              [rate_series(names.get(key, key), at(lambda row, key=key: row["by_opponent"].get(key)))
               for key in opponents], unit="%", x=x, domain=[0, 100]),
        chart("Win rate per side", "lines", [series(side, at(lambda row, side=side: side_rate(row, side)))
                                             for side in eval_pool.SIDES], unit="%", x=x, domain=[0, 100]),
        chart("Raze hero-hit rate", "lines",
              [series(hero, [None if rate is None else 100 * rate for rate in
                             at(lambda row, hero=hero: row["raze_hero_hit_rate"][hero]["rate"])])
               for hero in ("own", "enemy")], unit="%", x=x),
        chart("Losses by reason", "bars",
              [series(reason, at(lambda row, reason=reason: row["end_reasons"]["loss"][reason]))
               for reason in eval_pool.END_REASONS], unit="games", x=x),
    ]
    for field in eval_pool.LEAD_FIELDS:
        charts.append(chart(f"Early {field} lead", "lines",
                            [series(minute, at(lambda row, minute=minute: row["leads"].get(minute, {}).get(field)))
                             for minute in eval_pool.MINUTES], x=x,
                            note="Own minus enemy at the game minute; mean over pool games."))
    return charts


def dashboard_data(report, model, window, refresh):
    records = model.records()
    return {"report": {key: report[key] for key in ("source", "campaign_id", "phase", "total_updates", "last_update",
                                                    "games", "overall", "recent", "timing", "processes",
                                                    "by_opponent", "blocks", "in_flight_games")},
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "refresh": refresh,
            "x": [record["update"] for record in records],
            "starts": [start for start in model.process_starts if start > 1],
            "transitions": report["environments"]["transitions"],
            "sections": dashboard_sections(records, window) if records else []}


def write_html(path, data):
    """Writes the dashboard atomically so a refreshing browser never reads a partial file."""
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    refresh = f'<meta http-equiv="refresh" content="{int(data["refresh"])}">' if data["refresh"] else ""
    template = io.read_bytes(Path(__file__).absolute().with_name("train_dashboard.html"), 1024 * 1024).decode()
    html = template.replace("<!--REFRESH-->", refresh).replace("/*DATA*/", payload)
    path = Path(os.path.abspath(path))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, path)


def add_arguments(parser):
    parser.add_argument("target", type=Path, help="campaign directory, log directory, or log file (.gz allowed)")
    parser.add_argument("--block", type=int, default=10, help="updates per text/JSON block")
    parser.add_argument("--window", type=int, default=10, help="updates per rolling win-rate window")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--html", type=Path, help="write a self-contained dashboard here")
    parser.add_argument("--refresh", type=int, help="dashboard auto-reload period in seconds")
    parser.add_argument("--follow", action="store_true",
                        help="regenerate the dashboard every --refresh seconds while the campaign runs")


def main(options):
    io.bounded_integer(options.window, "window", 1, 1000)
    if options.refresh is not None:
        io.bounded_integer(options.refresh, "refresh", 5, 3600)
    if options.follow and (options.html is None or options.refresh is None):
        raise ValueError("--follow requires --html and --refresh")
    # A week of follow-up regenerations at the shortest period bounds the loop.
    for _ in range(7 * 24 * 3600 // 5):
        report, model = build_report(options.target, options.block)
        if options.html is not None:
            data = dashboard_data(report, model, options.window, options.refresh)
            data["sections"] = evaluation_sections(options.target) + data["sections"]
            write_html(options.html, data)
        if not options.follow or report["phase"] != "running":
            break
        time.sleep(options.refresh)
    if options.html is None:
        print(json.dumps(report, sort_keys=True) if options.as_json else format_report(report))
    else:
        print(json.dumps({"html": str(options.html), "updates": report["last_update"]}, sort_keys=True))
    return 0
