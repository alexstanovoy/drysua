"""Bounded CLI-contract/relay mocks: no Rust execution, wall-clock waits, or matches."""

from pathlib import Path
import socket
import struct
import unittest
from unittest.mock import Mock, patch

import release_crossplay as crossplay
import release_wire as wire
from test_release_crossplay import client
from test_release_wire import BOTH, events, frame, integer, observer, snapshot_with_effects


def scheduled_commands(registry):
    simulator = registry.get("simulator_commit", wire.HISTORICAL_SIMULATOR)
    relays = [Mock(observed=dict(slot=slot, winner="Neutral")) for slot in (0, 1)]
    bots = [dict(binary=Path("bot"), policy="teacher")] * 2
    with patch.object(crossplay, "validate_runtime_contract", return_value=simulator), \
            patch.object(crossplay, "launch", return_value=Mock()), \
            patch.object(crossplay, "server_port", return_value=4455), \
            patch.object(crossplay, "Relay", side_effect=relays) as relay, \
            patch.object(crossplay, "stop_game"), patch.object(crossplay, "await_cap_events"), \
            patch.object(crossplay.time, "monotonic", return_value=0), \
            patch.object(crossplay, "collect", return_value=[client("Radiant"), client("Dire")]):
        game = crossplay.execute_game(Path("output"), Path("server"), bots, 7, registry)
    return game["commands"][1:], relay.call_args_list


class SnapshotLimitCli:
    """The current seat.rs snapshot-limit and completed-Events-before-MatchOver contract."""

    def __init__(self, command):
        assert command[1] == "play"
        self.limit = int(command[command.index("--limit") + 1])
        self.buffer = bytearray()
        self.outgoing = b""
        self.received = []
        self.pending = None
        self.exited = False
        self.winner = None

    def receive(self, data):
        if self.exited:
            raise BrokenPipeError("mock CLI already exited on snapshot limit")
        self.buffer.extend(data)
        assert len(self.buffer) <= 1024
        for _ in range(4):
            if len(self.buffer) < 4:
                return
            length = struct.unpack_from("<I", self.buffer)[0]
            if len(self.buffer) < length + 4:
                return
            payload = bytes(self.buffer[4:length + 4])
            del self.buffer[:length + 4]
            self.message(payload)
        raise AssertionError("terminal fixture frame batch exceeded")

    def message(self, payload):
        kind, offset = wire.varint(payload, 0)
        self.received.append(kind)
        if kind == 3:
            tick, _ = wire.varint(payload, offset)
            self.pending = tick
            if tick >= self.limit:
                self.exited = True
                self.outgoing = frame(b"\x04" + integer(tick))
        elif kind == 4:
            tick = wire.verify_events(payload)
            assert tick == self.pending
            self.pending = None
            self.outgoing = frame(b"\x04" + integer(tick))
        elif kind == 7:
            assert self.pending is None, "CLI requires Events before MatchOver"
            winner, _ = wire.varint(payload, offset)
            assert winner == 2
            self.winner, self.exited = "Neutral", True
        else:
            raise AssertionError("unexpected terminal fixture message")

    def send(self, maximum):
        assert maximum == wire.READ_CHUNK
        if self.outgoing:
            output, self.outgoing = self.outgoing, b""
            return output
        assert self.exited, "mock CLI cannot send EOF before exiting"
        return b""


def terminal_chunks(slot):
    # Replace the fixture's snapshot tick/viewer; no 27k-tick simulation is necessary.
    snapshot = b"\x03" + integer(27900) + bytes([1, slot]) + snapshot_with_effects()[5:]
    over = frame(b"\x07\x02" + integer(27900) + b"\x02" + bytes(9) + b"\x01" + bytes(8))
    return [frame(snapshot), frame(events(BOTH, tick=27900)), over[:6], over[6:]]


def drain_terminal(command, slot=0, client_eof_first=False, truncate_match_over=False):
    relay = observer(27900)
    relay.observed.update(slot=slot, last_snapshot=27899)
    relay.stop = Mock(is_set=Mock(return_value=False))
    peer = SnapshotLimitCli(command)
    client_socket = Mock(sendall=Mock(side_effect=peer.receive), recv=Mock(side_effect=peer.send))
    server_socket = Mock()
    chunks = terminal_chunks(slot)
    if truncate_match_over:
        chunks[-1] = chunks[-1][:-1]

    def server_receive(maximum):
        assert maximum == wire.READ_CHUNK
        if len(chunks) <= 2:
            assert relay.observed["cap_ack"], "ACK must be held before later MatchOver chunks"
            assert relay.observed["cap_events"]
            server_socket.sendall.assert_not_called()
            if chunks:
                assert not peer.exited, "CLI must stay alive until the final MatchOver chunk"
        return chunks.pop(0) if chunks else b""

    server_socket.recv.side_effect = server_receive
    schedule = [server_socket, server_socket, client_socket, server_socket, server_socket]
    schedule += [client_socket, server_socket] if client_eof_first else [server_socket, client_socket]

    def readable(readers, writers, errors, timeout):
        assert readers, "relay must return after both EOFs, not poll an empty reader set"
        assert timeout == 0.1
        assert schedule, "terminal drain exceeded seven I/O steps"
        source = schedule.pop(0)
        assert source in readers
        return [source], [], []

    with patch.object(wire.select, "select", side_effect=readable):
        relay.pump(client_socket, server_socket)
    assert not schedule
    assert not chunks
    return relay, peer, client_socket, server_socket


class TerminalTests(unittest.TestCase):
    def test_execute_full_native_cap_changes_only_bot_safety_limit(self):
        registry = crossplay.current_map2_registry("a" * 64)
        before = registry.copy()
        commands, relays = scheduled_commands(registry)
        for command in commands:
            self.assertEqual(command[command.index("--limit") + 1], "27901")
        self.assertEqual([call.args[2] for call in relays], [27900, 27900])
        self.assertEqual(registry, before)

    def test_short_current_caps_and_all_historical_caps_keep_exact_bot_and_relay_limit(self):
        cases = [(2, wire.CURRENT_SIMULATOR, tick) for tick in (1, 100, 27899)]
        cases += [(0, wire.HISTORICAL_SIMULATOR, 108900), (1, wire.HISTORICAL_SIMULATOR, 30000),
                  (1, wire.HISTORICAL_SIMULATOR, 27900)]
        for map_id, simulator, tick in cases:
            registry = dict(map=map_id, simulator_commit=simulator, tick_limit=tick, process_timeout_seconds=180)
            with self.subTest(map=map_id, tick=tick):
                commands, relays = scheduled_commands(registry)
                self.assertEqual([command[command.index("--limit") + 1] for command in commands], [str(tick)] * 2)
                self.assertEqual([call.args[2] for call in relays], [tick] * 2)

    def test_old_27900_cli_limit_exits_on_snapshot_without_consuming_final_events(self):
        command = crossplay.bot_command(dict(binary="bot", policy="teacher"), "local", 0, 27900)
        peer = SnapshotLimitCli(command)
        peer.receive(terminal_chunks(0)[0])
        self.assertTrue(peer.exited)
        self.assertEqual(peer.received, [3])
        self.assertIsNone(peer.winner)
        with self.assertRaisesRegex(BrokenPipeError, "already exited on snapshot limit"):
            peer.receive(terminal_chunks(0)[1])

    def test_scheduled_native_cap_delivers_events_and_later_neutral_before_cli_exit(self):
        commands, _ = scheduled_commands(crossplay.current_map2_registry("a" * 64))
        clients = []
        for slot, command in enumerate(commands):
            relay, peer, _, server_socket = drain_terminal(command, slot, client_eof_first=True)
            self.assertEqual(peer.received, [3, 4, 7])
            self.assertTrue(peer.exited)
            self.assertEqual(peer.winner, "Neutral")
            self.assertTrue(relay.observed["cap_ack"])
            self.assertTrue(relay.observed["cap_events"])
            server_socket.sendall.assert_not_called()
            record = client(crossplay.SIDES[slot], peer.winner)
            record.update(wire=relay.observed, stdout=record["stdout"].replace("100 ticks", "27900 ticks"))
            clients.append(record)
        self.assertEqual(crossplay.validate_game(clients, 0, False, 27900), ("draw", []))

    def test_server_eof_then_cap_client_eof_finishes_without_empty_reader_poll(self):
        command = crossplay.bot_command(dict(binary="bot", policy="teacher"), "local", 0, 27901)
        relay, peer, client_socket, _ = drain_terminal(command)
        self.assertEqual(relay.observed["winner"], "Neutral")
        self.assertEqual(peer.received, [3, 4, 7])
        client_socket.shutdown.assert_called_once_with(socket.SHUT_WR)

    def test_native_safety_limit_does_not_allow_snapshot_27901_through_relay(self):
        relay = observer(27900)
        with self.assertRaisesRegex(ValueError, "server exceeded tick limit"):
            relay.observe(frame(b"\x03" + integer(27901) + bytes([1, 0])))
        self.assertEqual(relay.observed["last_snapshot"], 27900)

    def test_truncated_late_matchover_after_held_ack_fails_instead_of_becoming_draw(self):
        command = crossplay.bot_command(dict(binary="bot", policy="teacher"), "local", 0, 27901)
        with self.assertRaisesRegex(ValueError, "truncated server frame"):
            drain_terminal(command, truncate_match_over=True)


if __name__ == "__main__":
    unittest.main()
