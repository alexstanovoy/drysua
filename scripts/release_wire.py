"""Bounded TCP relay observing the pinned bota postcard protocol, not bot claims."""

import select
import socket
import struct
import threading


def varint(payload, offset):
    value = 0
    for index in range(10):
        if offset + index >= len(payload):
            raise ValueError("truncated postcard integer")
        byte = payload[offset + index]
        value |= (byte & 127) << (7 * index)
        if byte < 128:
            return value, offset + index + 1
    raise ValueError("oversized postcard integer")


class Relay:
    """One bot connection; Welcome gates the next launch without a seat race."""

    def __init__(self, port, timeout, tick_limit, expected_map=None, expected_seed=None,
                 byte_limit=512 * 1024 * 1024):
        self.port = port
        self.timeout = timeout
        self.tick_limit = tick_limit
        self.expected_map = expected_map
        self.expected_seed = expected_seed
        self.byte_limit = byte_limit
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(timeout)
        self.address = f"127.0.0.1:{self.listener.getsockname()[1]}"
        self.welcomed = threading.Event()
        self.stop = threading.Event()
        self.observed = dict(slot=None, winner=None, rejected=0, errors=[], last_snapshot=0,
                             cap_ack=False, cap_events=False, map=None)
        self.buffer = bytearray()
        self.client_buffer = bytearray()
        self.client_bytes = self.client_frames = 0
        self.frames = 0
        self.bytes = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def observe(self, data):
        self.bytes += len(data)
        if self.bytes > getattr(self, "byte_limit", 512 * 1024 * 1024):
            raise ValueError("wire byte limit exceeded")
        self.buffer.extend(data)
        for _ in range(16385):
            if len(self.buffer) < 4:
                return
            length = struct.unpack_from("<I", self.buffer)[0]
            if not 1 <= length <= 4 * 1024 * 1024:
                raise ValueError("invalid wire frame length")
            if len(self.buffer) < length + 4:
                return
            payload = bytes(self.buffer[4:4 + length])
            del self.buffer[:4 + length]
            self.frames += 1
            if self.frames > self.tick_limit * 10 + 1000:
                raise ValueError("wire frame limit exceeded")
            self.observe_message(payload)
        raise ValueError("wire batch limit exceeded")

    def observe_message(self, payload):
        kind, offset = varint(payload, 0)
        if kind == 0:  # Welcome proves the first bot was seated before launch two.
            _, offset = varint(payload, offset)
            present, offset = varint(payload, offset)
            slot, offset = varint(payload, offset)
            _, offset = varint(payload, offset)
            mode, offset = varint(payload, offset)
            if present != 1 or slot not in (0, 1) or mode != 1:
                raise ValueError("invalid seat or non-lockstep Welcome")
            if self.observed["slot"] is not None or offset != len(payload):
                raise ValueError("duplicate or malformed Welcome")
            self.observed["slot"] = slot
            self.welcomed.set()
        elif kind == 2:
            match_id, offset = varint(payload, offset)
            map_id, offset = varint(payload, offset)
            tick_rate, _ = varint(payload, offset)
            if self.observed.get("map") is not None or tick_rate != 30:
                raise ValueError("duplicate MatchStart or wrong tick rate")
            if self.expected_map is not None and map_id != self.expected_map:
                raise ValueError("MatchStart map mismatch")
            if self.expected_seed is not None and match_id != self.expected_seed:
                raise ValueError("MatchStart seed mismatch")
            self.observed["map"] = map_id
            self.observed["match_id"] = match_id
        elif kind == 3:
            tick, offset = varint(payload, offset)
            if tick > self.tick_limit:
                raise ValueError("server exceeded tick limit")
            present, offset = varint(payload, offset)
            viewer, _ = varint(payload, offset)
            if present != 1 or viewer != self.observed["slot"]:
                raise ValueError("snapshot identity mismatch")
            if tick <= self.observed.get("last_snapshot", 0):
                raise ValueError("snapshot tick did not advance")
            self.observed["last_snapshot"] = tick
        elif kind == 4:
            tick, _ = varint(payload, offset)
            if tick != self.observed.get("last_snapshot"):
                raise ValueError("Events snapshot tick mismatch")
            if tick == self.tick_limit:
                if self.observed.get("cap_events"):
                    raise ValueError("duplicate cap Events")
                self.observed["cap_events"] = True
        elif kind == 5:
            self.observed["rejected"] += 1
        elif kind == 7:
            winner, offset = varint(payload, offset)
            if winner not in (0, 1, 2) or self.observed["winner"] is not None:
                raise ValueError("invalid or duplicate MatchOver")
            duration, offset = varint(payload, offset)
            count, offset = varint(payload, offset)
            if not 0 < duration <= self.tick_limit or count != 2:
                raise ValueError("invalid MatchOver duration or seat count")
            for index in range(count):
                slot, offset = varint(payload, offset)
                if slot != index:
                    raise ValueError("MatchOver seat identity mismatch")
                for _ in range(8):
                    _, offset = varint(payload, offset)
            if offset != len(payload):
                raise ValueError("trailing MatchOver bytes")
            self.observed["winner"] = ("Radiant", "Dire", "Neutral")[winner]
            self.observed["duration"] = duration
        elif kind > 8:
            raise ValueError("unknown server message")

    def filter_client(self, data):
        self.client_bytes += len(data)
        if self.client_bytes > 64 * 1024 * 1024:
            raise ValueError("client wire byte limit exceeded")
        self.client_buffer.extend(data)
        output = bytearray()
        for _ in range(16385):
            if len(self.client_buffer) < 4:
                return output
            length = struct.unpack_from("<I", self.client_buffer)[0]
            if not 1 <= length <= 4 * 1024 * 1024:
                raise ValueError("invalid client frame length")
            if len(self.client_buffer) < length + 4:
                return output
            frame = bytes(self.client_buffer[:length + 4])
            del self.client_buffer[:length + 4]
            self.client_frames += 1
            if self.client_frames > self.tick_limit * 4 + 1000:
                raise ValueError("client wire frame limit exceeded")
            kind, offset = varint(frame, 4)
            if kind == 4:
                tick, offset = varint(frame, offset)
                if offset != len(frame) or tick > self.observed.get("last_snapshot", 0):
                    raise ValueError("ACK exceeds observed snapshot or malformed ACK")
                if tick == self.tick_limit:
                    if self.observed.get("cap_ack"):
                        raise ValueError("duplicate cap ACK")
                    # Forwarding this ACK permits cap+1 before the client EOF reaches the server.
                    self.observed["cap_ack"] = True
                    continue
            output.extend(frame)
        raise ValueError("client wire batch limit exceeded")

    def run(self):
        try:
            with self.listener:
                client, _ = self.listener.accept()
            with client, socket.create_connection(("127.0.0.1", self.port), 5) as server:
                client.settimeout(5)
                server.settimeout(5)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.pump(client, server)
        except (OSError, ValueError) as error:
            if not self.stop.is_set():
                self.observed["errors"].append(str(error))
        finally:
            self.welcomed.set()

    def pump(self, client, server):
        readers = [client, server]
        writable = True
        client_open = True
        for _ in range(self.tick_limit * 40 + 4000):
            if self.stop.is_set():
                return
            readable, _, _ = select.select(readers, [], [], 0.1)
            for source in readable:
                try:
                    data = source.recv(65536)
                except ConnectionResetError:
                    if source is not client or not self.observed.get("cap_ack"):
                        raise
                    data = b""
                if not data:
                    if source is client:
                        if getattr(self, "client_buffer", None):
                            raise ValueError("truncated client frame")
                        if self.observed.get("cap_ack"):
                            readers.remove(client)
                            client_open = False
                            continue
                        return
                    if self.buffer:
                        raise ValueError("truncated server frame")
                    if self.observed["winner"] is None and not (
                            self.observed.get("cap_ack") and self.observed.get("cap_events")):
                        raise ValueError("server closed before MatchOver")
                    # Closing with unread final ACKs sends RST instead of delivering MatchOver.
                    readers.remove(server)
                    writable = False
                    if client_open:
                        client.shutdown(socket.SHUT_WR)
                    else:
                        return
                elif source is server:
                    self.observe(data)
                    if client_open:
                        try:
                            client.sendall(data)
                        except (BrokenPipeError, ConnectionResetError):
                            if not self.observed.get("cap_ack"):
                                raise
                            client_open = False
                elif writable:
                    try:
                        filtered = self.filter_client(data)
                        if filtered:
                            server.sendall(filtered)
                    except (BrokenPipeError, ConnectionResetError):
                        # Drain the server's terminal frame before deciding whether EOF is valid.
                        writable = False
        raise ValueError("relay iteration limit exceeded")

    def close(self):
        self.stop.set()
        self.listener.close()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            self.observed["errors"].append("relay failed to stop")
