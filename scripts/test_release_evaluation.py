"""Deterministic current Map2 evaluator contracts, separate from historical gates."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import release_crossplay as crossplay
from release_build import SIMULATOR
from test_release_crossplay import client


CURRENT = "037c6a2f8e5383beae9eea6da8cbbb1678f7b718"
LOG_LIMIT = 1024 * 1024


def performance_line(slot=0, scope="window", reason="periodic"):
    updates = 27900 if scope == "total" else 300
    progress = updates - 1 if scope == "total" else updates
    factor = "0.999" if scope == "total" else "1.000"
    line = (f"level=INFO event=live_performance scope={scope} reason={reason} updates={updates} "
            f"progress_ticks={progress} elapsed_ns={updates // 30 * 10**9} updates_per_second=30.000 "
            f"realtime_factor={factor} tick_rate=30 budget_ns=33333333 compute_overruns=0 "
            "service_overruns=0 percentiles=log2_upper_bounds")
    for name in ("compute", "receive_wait", "decision", "order_send", "ack_send"):
        count = updates // 3 if name == "decision" else updates
        line += (f" {name}_count={count} {name}_total_ns={count * 1000000} {name}_p50_upper_ns=1000000"
                 f" {name}_p95_upper_ns=1000000 {name}_max_ns=1000000")
    return (line + f" pending_update=false saturated=false slot={slot} policy=teacher mode=lockstep "
            "receive_wait_scope=socket_read timing_valid=true dropped_logs=0\n")


def map2_client(slot):
    side = ("Radiant", "Dire")[slot]
    record = client(side, "Neutral")
    start = (f"level=INFO event=live_performance_start slot={slot} policy=teacher mode=lockstep "
             "tick_rate=30 report_every=300 debug_every=300 debug_limit=32 receive_wait_scope=socket_read "
             "compute_scope=internal_elapsed_excluding_receive_and_send dropped_logs=0\n")
    debug = (f"level=DEBUG event=live_decision slot={slot} policy=teacher tick=300 decision_ns=1000000 "
             "order_sent=false order_send_ns=unknown ack_send_ns=1000 dropped_logs=0\n")
    first = performance_line(slot).replace("progress_ticks=300", "progress_ticks=299")
    first = first.replace("realtime_factor=1.000", "realtime_factor=0.996")
    record["stdout"] = (start + first + performance_line(slot) * (27900 // 300 - 1) + debug
                        + performance_line(slot, "total", "match_over")
                        + record["stdout"].replace("played 100 ticks", "played 27900 ticks")
                        .replace("10 decisions, 5 orders", "9300 decisions, 3100 orders"))
    record["wire"].update(last_snapshot=27900, duration=27900)
    return record


class ClientLogTests(unittest.TestCase):
    def test_known_config_defaulted_info_is_accepted_without_hiding_disabled_or_unknown_diagnostics(self):
        summary = client("Radiant", "Neutral")["stdout"]
        reasons = ("DRYSUA_PERF_DEBUG_EVERY must be Unicode digits",
                   "DRYSUA_PERF_DEBUG_EVERY must be an integer in 0..=4294967295")
        for reason in reasons:
            line = f'level=INFO event=live_performance_config_defaulted reason="{reason}" dropped_logs=0\n'
            self.assertEqual(crossplay.outcome_summary(line + summary, 0).group(4), "Neutral")
        for event, reason in (("live_performance_config_defaulted", "unknown error"),
                              ("live_performance_disabled", "performance clock must not regress")):
            line = f'level=INFO event={event} reason="{reason}" dropped_logs=0\n'
            with self.assertRaisesRegex(ValueError, "invalid client telemetry"):
                crossplay.outcome_summary(line + summary, 0)

    def test_async_writer_suffix_is_accepted_on_start_info_warn_and_debug_records(self):
        text = map2_client(0)["stdout"]
        lines = text.splitlines()
        records = [lines[0], lines[1], lines[1].replace("level=INFO", "level=WARN"), lines[-3]]
        summary = lines[-1]
        for record in records:
            for dropped in (0, 7, 2**64 - 1):
                with self.subTest(record=record[:65], dropped=dropped):
                    output = record.replace("dropped_logs=0", f"dropped_logs={dropped}") + "\n" + summary
                    self.assertEqual(crossplay.outcome_summary(output, 0).group(4), "Neutral")

    def test_async_writer_suffix_rejects_malformed_duplicate_and_overflow_counters(self):
        line = performance_line()
        for value, message in (("", "invalid client telemetry"), ("-1", "invalid client telemetry"),
                               ("no", "invalid client telemetry"), ("01", "invalid client telemetry"),
                               ("0 dropped_logs=0", "invalid client telemetry"),
                               ("0 unexpected=true", "invalid client telemetry"),
                               (str(2**64), "invalid client dropped_logs counter")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                crossplay.outcome_summary(line.replace("dropped_logs=0", f"dropped_logs={value}"), 0)

    def test_synchronous_telemetry_without_async_suffix_remains_accepted(self):
        record = map2_client(0)
        output = record["stdout"].replace(" dropped_logs=0", "")
        self.assertEqual(crossplay.outcome_summary(output, 0).group(4), "Neutral")

    def test_extra_or_missing_clients_are_errors_not_an_assertion_or_draw(self):
        for clients in ([map2_client(0), map2_client(1), map2_client(0)], [map2_client(0)]):
            self.assertEqual(crossplay.validate_game(clients, 0, False, 27900),
                             ("error", ["missing client"]))

    def test_full_map2_perf_log_above_64k_is_preserved_by_monitor_and_reader(self):
        expected = map2_client(0)["stdout"]
        self.assertGreater(len(expected.encode()), 65536)
        self.assertLess(len(expected.encode()), LOG_LIMIT)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "client-0.log"
            path.write_text(expected)
            crossplay.check_resources(directory, [path])
            clients = crossplay.collect([Mock(), Mock(returncode=0, pid=101)], [path],
                                       [Mock(observed=map2_client(0)["wire"])], set())
        self.assertEqual(clients[0]["stdout"], expected)

    def test_exact_new_bound_accepted_but_one_extra_byte_fails_monitor_and_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "client-0.log"
            path.write_bytes(b"x" * LOG_LIMIT)
            crossplay.check_resources(directory, [path])
            self.assertEqual(len(crossplay.read_client_output(path)), LOG_LIMIT)
            path.write_bytes(b"x" * (LOG_LIMIT + 1))
            with self.assertRaisesRegex(ValueError, "client output limit exceeded"):
                crossplay.check_resources(directory, [path])
            clients = crossplay.collect([Mock(), Mock(returncode=0, pid=101)], [path],
                                       [Mock(observed=map2_client(0)["wire"])], set())
        self.assertIn("client output limit exceeded", clients[0]["output_errors"])
        self.assertEqual(clients[0]["stdout"], "")

    def test_reader_is_bounded_even_if_file_grows_after_stat(self):
        path = MagicMock()
        path.stat.return_value.st_size = 1
        stream = io.BytesIO(b"x" * (LOG_LIMIT + 2))
        path.open.return_value.__enter__.return_value = stream
        with self.assertRaisesRegex(ValueError, "client output limit exceeded"):
            crossplay.read_client_output(path)
        self.assertEqual(stream.tell(), LOG_LIMIT + 1)

    def test_invalid_utf8_is_a_collected_error_not_a_crash_or_silent_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "client.log"
            path.write_bytes(b"\xff")
            clients = crossplay.collect([Mock(), Mock(returncode=0, pid=101)], [path],
                                       [Mock(observed=map2_client(0)["wire"])], set())
        self.assertIn("client output is not UTF-8", clients[0]["output_errors"])

    def test_full_telemetry_logs_with_native_draw_are_not_a_loss_or_opponent_win(self):
        result = crossplay.validate_game([map2_client(0), map2_client(1)], 0, False, 27900)
        self.assertEqual(result, ("draw", []))

    def test_warn_performance_report_is_retained_and_not_a_process_failure(self):
        clients = [map2_client(0), map2_client(1)]
        clients[0]["stdout"] = clients[0]["stdout"].replace(
            "level=INFO event=live_performance scope", "level=WARN event=live_performance scope")
        self.assertEqual(crossplay.validate_game(clients, 0, False, 27900), ("draw", []))

    def test_unknown_malformed_duplicate_or_error_output_cannot_hide_behind_summary(self):
        for text in ("drysua: decode error\n", "level=ERROR event=live_performance broken=true\n",
                     "level=INFO event=unknown value=1\n", "level=INFO event=live_performance\n",
                     client("Radiant", "Neutral")["stdout"],
                     performance_line().replace("timing_valid=true", "timing_valid=false"),
                     performance_line(slot=1)):
            with self.subTest(text=text[:60]):
                clients = [map2_client(0), map2_client(1)]
                clients[0]["stdout"] = text + clients[0]["stdout"]
                self.assertEqual(crossplay.validate_game(clients, 0, False, 27900)[0], "error")

    def test_in_memory_log_over_bound_and_collected_errors_cannot_be_a_draw(self):
        clients = [map2_client(0), map2_client(1)]
        clients[0]["stdout"] = "x" * (LOG_LIMIT + 1)
        self.assertIn("client output limit exceeded", crossplay.validate_game(clients, 0, False, 27900)[1])
        clients[0] = map2_client(0)
        clients[0]["output_errors"] = ["client output is not UTF-8"]
        self.assertIn("client output is not UTF-8", crossplay.validate_game(clients, 0, False, 27900)[1])


class RuntimeContractTests(unittest.TestCase):
    def test_mismatched_execute_game_never_launches_a_process_or_relay(self):
        registry = dict(map=2, simulator_commit=SIMULATOR)
        with patch.object(crossplay, "launch") as launch, patch.object(crossplay, "Relay") as relay:
            with self.assertRaisesRegex(ValueError, "simulator/map contract mismatch"):
                crossplay.execute_game(Path("output"), Path("server"), [], 7, registry)
        launch.assert_not_called()
        relay.assert_not_called()

    def test_execute_game_passes_explicit_current_and_historical_pins_to_relay(self):
        for map_id, simulator, tick_limit in ((0, SIMULATOR, 108900), (1, SIMULATOR, 30000),
                                             (2, CURRENT, 27900)):
            with self.subTest(map_id=map_id):
                registry = dict(map=map_id, simulator_commit=simulator,
                                tick_limit=tick_limit, process_timeout_seconds=180)
                relays = [Mock(observed=dict(slot=slot, winner="Neutral")) for slot in (0, 1)]
                bots = [dict(binary=Path("bot"), policy="teacher")] * 2
                with patch.object(crossplay, "validate_runtime_contract", return_value=simulator), \
                        patch.object(crossplay, "launch", return_value=Mock()), \
                        patch.object(crossplay, "server_port", return_value=4455), \
                        patch.object(crossplay, "Relay", side_effect=relays) as relay, \
                        patch.object(crossplay, "stop_game"), \
                        patch.object(crossplay, "await_cap_events"), \
                        patch.object(crossplay.time, "monotonic", return_value=0), \
                        patch.object(crossplay, "collect", return_value=[map2_client(0), map2_client(1)]):
                    game = crossplay.execute_game(Path("output"), Path("server"), bots, 7, registry)
                self.assertEqual(game["result"], "draw")
                self.assertEqual([call.kwargs["simulator_commit"] for call in relay.call_args_list],
                                 [simulator, simulator])

    def test_map2_helper_is_explicit_current_and_includes_pregame_in_cap(self):
        registry = crossplay.current_map2_registry("a" * 64)
        self.assertEqual(registry["map"], 2)
        self.assertEqual(registry["tick_limit"], 27000 + 900)
        self.assertEqual(registry["simulator_commit"], CURRENT)
        self.assertNotIn("gate", registry)
        self.assertNotIn("cpu_threads", registry)

    def test_historical_registry_is_never_relabelled_current(self):
        for registry in (dict(map=1, simulator_commit=SIMULATOR), dict(map=0)):
            before = registry.copy()
            self.assertEqual(crossplay.validate_runtime_contract(Path("server"), [], registry), SIMULATOR)
            self.assertEqual(registry, before)

    def test_historical_and_current_simulator_mismatch_fails_before_launch(self):
        for registry, bots in ((dict(map=2, simulator_commit=SIMULATOR), []),
                               (dict(map=1, simulator_commit=CURRENT), []),
                               (dict(map=0), [dict(simulator_commit=CURRENT)])):
            with self.assertRaisesRegex(ValueError, "simulator.*mismatch"):
                crossplay.validate_runtime_contract(Path("server"), bots, registry)

    def test_current_map2_requires_digest_bound_runtime_attestations(self):
        registry = crossplay.current_map2_registry("a" * 64)
        bot = dict(binary=Path("bot"), simulator_commit=CURRENT, sha256="b" * 64)
        with patch.object(crossplay, "digest", side_effect=["a" * 64, "b" * 64, "b" * 64]):
            self.assertEqual(crossplay.validate_runtime_contract(Path("server"), [bot, bot], registry), CURRENT)
        for invalid, message in ((dict(binary=Path("bot")), "runtime attestation"),
                                 (dict(bot, simulator_commit=SIMULATOR), "simulator.*mismatch"),
                                 (dict(bot, sha256="c" * 64), "runtime SHA256 mismatch")):
            with patch.object(crossplay, "digest", side_effect=["a" * 64, "b" * 64]):
                with self.assertRaisesRegex(ValueError, message):
                    crossplay.validate_runtime_contract(Path("server"), [invalid, bot], registry)

    def test_current_map2_server_hash_cap_and_timeout_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "server SHA256"):
            crossplay.current_map2_registry("bad")
        for timeout in (0, 601, True):
            with self.assertRaisesRegex(ValueError, "process timeout"):
                crossplay.current_map2_registry("a" * 64, process_timeout_seconds=timeout)
        registry = crossplay.current_map2_registry("a" * 64)
        registry["tick_limit"] = 27901
        with self.assertRaisesRegex(ValueError, "Map2 tick limit"):
            crossplay.validate_runtime_contract(Path("server"), [], registry)


if __name__ == "__main__":
    unittest.main()
