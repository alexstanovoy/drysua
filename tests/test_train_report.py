"""Report and dashboard contracts on small fixture logs of every campaign layout."""
import contextlib
import gzip
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train  # noqa: E402
import train_report  # noqa: E402

HEADER = "annealed: updates=4 games=2 parallel=2 generation_games=2 zero_updates=0 seed=1 opponent=Teacher\n"


def episode(outcome, opponent="Teacher", tick=1000, extra=""):
    return (f"episode: stream=0 map=2 opponent={opponent} tick={tick} outcome={outcome} "
            f"actions=[3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] raw_return=0.5{extra}\n")


def progress(update, durable=True, **fields):
    """The per-update statistics, then the checkpoint marker unless the trainer has not committed yet."""
    extra = "".join(f", {key.replace('_', ' ')} {value}" for key, value in fields.items())
    return (f"level=INFO event=training_update_timing update_index={update - 1} elapsed_ns=2000000000 "
            f"collection_ns=1500000000 samples=100\n"
            f"progress: update {update}, samples {update * 100}, policy loss -0.01, KL 0.002, KL stop false{extra}\n"
            + (f"checkpoint: update {update}\n" if durable else ""))


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, relative, text, compress=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with gzip.open(path, "wt") as target:
                target.write(text)
        else:
            path.write_text(text)
        return path

    def test_old_per_invocation_layout_counts_only_durable_updates(self):
        self.write("invocations/000001/payload.log", HEADER + episode("Win") + episode("Loss") + progress(1))
        # A failed invocation never committed; its games are replayed by the next one.
        self.write("invocations/000002/payload.log", HEADER + episode("Win") + episode("Win"))
        self.write("invocations/000003/payload.log", HEADER + episode("Draw") + episode("Loss") + progress(2))
        report, _ = train_report.build_report(self.root)
        self.assertEqual(report["per_update"], [{"update": 1, "wins": 1, "losses": 1, "draws": 0},
                                                {"update": 2, "wins": 0, "losses": 1, "draws": 1}])
        self.assertEqual(report["overall"]["games"], 4)

    def test_gzip_history_with_invocation_separators_and_a_running_tail(self):
        text = ("### invocation 000001\n" + HEADER + episode("Win") + episode("Win") + progress(1) +
                "### invocation 000002\n" + HEADER + episode("Loss") + episode("Win") + progress(2) +
                episode("Loss") + episode("Loss"))
        path = self.write("history/payload.log.gz", text, compress=True)
        report, _ = train_report.build_report(path)
        self.assertEqual((report["last_update"], report["overall"]["wins"], report["in_flight_games"]), (2, 3, 2))
        self.assertEqual(train_report.build_report(path.parent)[0]["last_update"], 2)

    def test_long_lived_session_log_groups_episodes_by_progress_line(self):
        self.write("sessions/0001/payload.log", HEADER + episode("Win") + episode("Win") + progress(1) +
                   episode("Loss") + episode("Loss") + progress(2) + episode("Win"))
        self.write("sessions/0002/payload.log", HEADER + episode("Draw") + episode("Win") + progress(3))
        report, model = train_report.build_report(self.root)
        self.assertEqual([entry["wins"] for entry in report["per_update"]], [2, 0, 1])
        self.assertEqual(model.process_starts, [1, 3])
        self.assertEqual(report["timing"]["seconds_per_update"], 2.0)

    def test_restart_replaces_updates_after_the_last_durable_checkpoint(self):
        # The first process logged updates 2 and 3 but only update 1 is durable; the resume logs them again.
        self.write("sessions/0001/payload.log", HEADER + episode("Win") + progress(1) + episode("Win") +
                   progress(2, durable=False) + episode("Win") + progress(3, durable=False) + episode("Win"))
        live, _ = train_report.build_report(self.root)
        self.assertEqual((live["last_update"], live["overall"]["wins"], live["in_flight_games"]), (3, 3, 1))
        self.write("sessions/0002/payload.log", HEADER + episode("Loss") + progress(2) + episode("Draw") + progress(3))
        report, model = train_report.build_report(self.root)
        self.assertEqual([(entry["update"], entry["wins"], entry["losses"], entry["draws"])
                          for entry in report["per_update"]], [(1, 1, 0, 0), (2, 0, 1, 0), (3, 0, 0, 1)])
        self.assertEqual((model.process_starts, report["in_flight_games"]), ([1, 2], 0))

    def test_new_numeric_fields_and_opponents_appear_in_the_dashboard_without_code_changes(self):
        self.write("payload.log", HEADER +
                   episode("Win", "Teacher", extra=" kills=2 end_reason=timeout") +
                   episode("Loss", "Pool3", extra=" kills=0 end_reason=throne") +
                   "level=INFO event=ppo_diagnostics clip_fraction=0.125 explained_variance=0.5\n" + progress(1))
        report, model = train_report.build_report(self.root)
        self.assertEqual(set(report["by_opponent"]), {"Teacher", "Pool3"})
        data = train_report.dashboard_data(report, model, window=10, refresh=None)
        titles = {section["title"]: [chart["title"] for chart in section["charts"]] for section in data["sections"]}
        self.assertIn("ppo_diagnostics.clip_fraction", titles["PPO"])
        self.assertIn("ppo_diagnostics.explained_variance", titles["PPO"])
        self.assertIn("episode.kills", titles["Episodes"])
        self.assertIn("episode end_reason share", titles["Episodes"])
        win_rate = data["sections"][0]["charts"][1]
        self.assertEqual([entry["name"] for entry in win_rate["series"]], ["all", "Pool3", "Teacher"])
        self.assertEqual(sum(1 for titles_list in titles.values() for title in titles_list
                             if title == "episode.kills"), 1)

    def test_html_is_self_contained_and_refreshes_only_when_asked(self):
        self.write("payload.log", HEADER + episode("Win") + episode("Loss") + progress(1))
        output = self.root / "dashboard.html"
        for refresh in (None, 30):
            with contextlib.redirect_stdout(io.StringIO()):
                train.main(["report", str(self.root / "payload.log"), "--html", str(output)] +
                           ([] if refresh is None else ["--refresh", str(refresh)]))
            html = output.read_text()
            self.assertEqual(re.findall(r"https?://(?!www\.w3\.org/2000/svg)", html), [])
            self.assertEqual('http-equiv="refresh" content="30"' in html, refresh is not None)
        embedded = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', html, re.S)
        self.assertEqual(json.loads(embedded.group(1))["x"], [1])

    def test_wilson_interval_stays_informative_at_the_boundaries(self):
        low, high = train_report.wilson(0, 10)
        self.assertEqual((low, round(high, 4)), (0.0, 0.2775))
        low, high = train_report.wilson(10, 10)
        self.assertEqual((round(low, 4), high), (0.7225, 1.0))


if __name__ == "__main__":
    unittest.main()
