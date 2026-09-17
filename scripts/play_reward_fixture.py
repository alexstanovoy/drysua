"""TEST FIXTURE ONLY: passive native Player, never a human-performance evaluation."""

import argparse
import json
import socket
import struct
import time

from play_admission import FRAME_LIMIT, SERVER_BYTE_LIMIT, verify_match_over, verify_welcome
from play_pacing import MAX_TICK, read_u32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", required=True)
    parser.add_argument("--slot", type=int, choices=(0, 1), required=True)
    ending = parser.add_mutually_exclusive_group()
    ending.add_argument("--complete", action="store_true", help="Wait for actual native MatchOver")
    ending.add_argument("--stop-tick", type=bounded_tick, default=60,
                        metavar="1..27900", help="Manual incomplete cut, no ACK for the last tick")
    arguments = parser.parse_args()
    host, port = arguments.addr.rsplit(":", 1)
    with socket.create_connection((host, int(port)), 5) as connection:
        connection.settimeout(10)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        report = play(connection, arguments.slot, None if arguments.complete else arguments.stop_tick)
    print(json.dumps(report), flush=True)


def bounded_tick(value):
    try:
        tick = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("fixture stop tick must be an integer in 1..27900") from error
    if not 1 <= tick <= MAX_TICK:
        raise argparse.ArgumentTypeError("fixture stop tick must be an integer in 1..27900")
    return tick


def play(connection, slot, stop_tick):
    send(connection, b"\x00\x00\x05human")
    state = {"fixture": "teacher-vs-passive-player; NOT human performance", "slot": slot,
             "ticks": 0, "snapshots": 0, "events": 0, "terminal": False, "winner": None}
    welcomed, started, first_snapshot = False, False, None
    deadline, received = time.monotonic() + 240, 0
    for _ in range(1_000_000):
        if time.monotonic() >= deadline:
            raise RuntimeError("native fixture exceeded 240 seconds")
        length = struct.unpack("<I", receive(connection, 4))[0]
        if not 1 <= length <= FRAME_LIMIT or received + length + 4 > SERVER_BYTE_LIMIT:
            raise ValueError("native fixture frame/stream limit exceeded")
        payload = receive(connection, length)
        received += length + 4
        kind, offset = read_u32(payload, 0)
        if kind == 0:
            if welcomed:
                raise ValueError("native fixture duplicate Welcome")
            verify_welcome(payload, slot, 1)
            welcomed = True
            send(connection, b"\x01\x02")
            send(connection, b"\x02\x01")
        elif kind == 2:
            if not welcomed or started:
                raise ValueError("native fixture invalid MatchStart ordering")
            started = True
        elif kind in (3, 4):
            if not started:
                raise ValueError("native fixture tick before MatchStart")
            tick, offset = read_u32(payload, offset)
            if kind == 3:
                if tick != state["snapshots"] + 1 or state["events"] != state["snapshots"]:
                    raise ValueError("native fixture missing Snapshot/Events pair")
                present, offset = read_u32(payload, offset)
                viewer, _ = read_u32(payload, offset)
                if present != 1 or viewer != slot:
                    raise ValueError("native fixture inconsistent viewer")
                first_snapshot = time.monotonic() if first_snapshot is None else first_snapshot
                state["snapshots"] = tick
                if tick != stop_tick:
                    send(connection, b"\x04" + encode_tick(tick))
            else:
                if tick != state["snapshots"] or tick != state["events"] + 1:
                    raise ValueError("native fixture missing or duplicate Events")
                state.update(events=tick, ticks=tick)
                if tick == stop_tick:
                    break
        elif kind == 7:
            verify_match_over(payload)
            winner, offset = read_u32(payload, offset)
            duration, _ = read_u32(payload, offset)
            if not state["ticks"] or duration != state["events"] or duration != state["snapshots"]:
                raise ValueError("native fixture MatchOver lacks complete final pair")
            state.update(terminal=True, winner=winner)
            break
        elif kind not in (1,):
            raise ValueError(f"unexpected native fixture server message {kind}")
    else:
        raise RuntimeError("native fixture frame count exceeded")
    state["elapsed_snapshot_seconds"] = time.monotonic() - first_snapshot
    return state


def receive(connection, count):
    output = bytearray()
    for _ in range(count):
        data = connection.recv(min(65536, count - len(output)))
        if not data:
            raise RuntimeError("native fixture EOF before requested boundary")
        output.extend(data)
        if len(output) == count:
            return bytes(output)
    raise RuntimeError("native fixture receive bound exceeded")


def send(connection, payload):
    connection.sendall(struct.pack("<I", len(payload)) + payload)


def encode_tick(value):
    if not 1 <= value <= MAX_TICK:
        raise ValueError("native fixture tick outside Map2 cap")
    output = bytearray()
    for _ in range(5):
        output.append((value & 127) | (128 if value > 127 else 0))
        value >>= 7
        if not value:
            return bytes(output)
    raise ValueError("native fixture tick exceeds u32")


if __name__ == "__main__":
    main()
