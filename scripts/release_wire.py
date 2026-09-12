"""Bounded TCP observer; current Events by default, historical decoding only by pin."""

import select
import socket
import struct
import threading


CURRENT_SIMULATOR = "037c6a2f8e5383beae9eea6da8cbbb1678f7b718"
HISTORICAL_SIMULATOR = "18db0f62d9a2b94e755c43fd29a959db204cc20b"
FRAME_LIMIT = 4 * 1024 * 1024
SERVER_BYTE_LIMIT = 512 * 1024 * 1024
CLIENT_BYTE_LIMIT = 64 * 1024 * 1024
READ_CHUNK = 65536
EVENT_LIMIT = FRAME_LIMIT // 3  # ItemBought is the shortest event: tag, u8 slot, u16 item.
assert READ_CHUNK < FRAME_LIMIT < CLIENT_BYTE_LIMIT < SERVER_BYTE_LIMIT
assert EVENT_LIMIT * 3 <= FRAME_LIMIT


def varint(payload, offset, bits=64):
    """Read one canonical postcard unsigned integer (not a raw u8)."""
    assert bits in (16, 32, 64)
    if not 0 <= offset <= len(payload):
        raise ValueError("invalid postcard offset")
    value = 0
    for index in range(10):
        if offset + index >= len(payload):
            raise ValueError("truncated postcard integer")
        byte = payload[offset + index]
        value |= (byte & 127) << (7 * index)
        if byte < 128:
            if value >= 1 << bits:
                raise ValueError(f"postcard integer exceeds u{bits}")
            if index and byte == 0:
                raise ValueError("noncanonical postcard integer")
            return value, offset + index + 1
    raise ValueError("oversized postcard integer")


class Postcard:
    """Scalar cursor over one bounded payload, without allocating event collections."""

    def __init__(self, payload, offset=0):
        if len(payload) > FRAME_LIMIT:
            raise ValueError("wire payload limit exceeded")
        if not 0 <= offset <= len(payload):
            raise ValueError("invalid postcard offset")
        self.payload, self.offset = payload, offset

    def integer(self, bits=32):
        value, self.offset = varint(self.payload, self.offset, bits)
        return value

    def byte(self):
        if self.offset >= len(self.payload):
            raise ValueError("truncated postcard byte")
        value = self.payload[self.offset]
        self.offset += 1
        return value

    def flag(self, name):
        value = self.byte()
        if value not in (0, 1):
            raise ValueError(f"invalid postcard {name}")
        return value

    def entity(self):
        self.integer()
        self.integer()

    def optional_entity(self):
        if self.flag("option"):
            self.entity()


def verify_event(cursor, simulator_commit):
    kind = cursor.integer()
    if kind in (0, 1):
        cursor.optional_entity()
        cursor.entity()
        cursor.integer()  # Signed i32 uses a zigzag-encoded u32.
        if kind == 0:
            if cursor.integer() > 2:
                raise ValueError("invalid damage kind")
            cursor.flag("bool")
        elif simulator_commit == CURRENT_SIMULATOR:
            cursor.integer()  # Mana is present even for an HP-only heal.
    elif kind == 2:
        cursor.entity()
        cursor.optional_entity()
        cursor.flag("bool")
        cursor.integer()
    elif kind == 3:
        cursor.entity()
        cursor.integer(16)
    elif kind == 4:
        cursor.entity()
        cursor.byte()
    elif kind == 5:
        cursor.byte()
        cursor.integer(16)
    elif kind == 6:
        cursor.entity()
        if cursor.integer() > 2:
            raise ValueError("invalid event team")
    else:
        raise ValueError("unknown event kind")


def verify_events(payload, simulator_commit=CURRENT_SIMULATOR):
    """Validate a complete ServerMsg::Events payload and return its u32 tick."""
    if simulator_commit not in (CURRENT_SIMULATOR, HISTORICAL_SIMULATOR):
        raise ValueError("unsupported simulator wire contract")
    cursor = Postcard(payload)
    if cursor.integer() != 4:
        raise ValueError("expected Events message")
    tick = cursor.integer()
    count = cursor.integer(64)
    if count > EVENT_LIMIT:
        raise ValueError("event count limit exceeded")
    for _ in range(count):
        verify_event(cursor, simulator_commit)
    if cursor.offset != len(payload):
        raise ValueError("trailing Events bytes")
    return tick


def verify_client_message(payload):
    """Validate the shared current/historical client schema; return kind and optional ACK tick."""
    cursor = Postcard(payload)
    kind, tick = cursor.integer(), None
    if kind == 0:
        if cursor.integer() > 2:
            raise ValueError("invalid client role")
        length = cursor.integer(64)
        end = cursor.offset + length
        if end > len(payload):
            raise ValueError("truncated client name")
        try:
            payload[cursor.offset:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid client name UTF-8") from error
        cursor.offset = end
    elif kind == 1:
        cursor.integer(16)
    elif kind == 2:
        cursor.flag("ready")
    elif kind == 3:
        cursor.integer()
        cursor.optional_entity()
        verify_client_order(cursor)
    elif kind == 4:
        tick = cursor.integer()
    elif kind == 5:
        if cursor.flag("option"):
            cursor.byte()
    else:
        raise ValueError("unknown client message")
    if cursor.offset != len(payload):
        if kind == 4:
            raise ValueError("ACK exceeds observed snapshot or malformed ACK")
        raise ValueError("trailing client message bytes")
    return kind, tick


def verify_client_order(cursor):
    kind = cursor.integer()
    if kind > 9:
        raise ValueError("unknown client order")
    if kind in (2, 3, 4, 7, 9):
        cursor.byte()
    if kind == 6:
        cursor.integer(16)
    if kind == 8:
        cursor.byte()
        cursor.byte()
    if kind <= 5:
        target = cursor.integer()
        if target in (1, 2):
            cursor.integer()
            cursor.integer()
        elif target != 0:
            raise ValueError("unknown client target")


class Relay:
    """One bot connection; Welcome gates the next launch without a seat race."""

    def __init__(self, port, timeout, tick_limit, expected_map=None, expected_seed=None,
                 byte_limit=SERVER_BYTE_LIMIT, simulator_commit=CURRENT_SIMULATOR):
        if simulator_commit not in (CURRENT_SIMULATOR, HISTORICAL_SIMULATOR):
            raise ValueError("unsupported simulator wire contract")
        self.simulator_commit = simulator_commit
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
                             cap_ack=False, cap_events=False, client_eof=False, map=None)
        self.buffer = bytearray()
        self.client_buffer = bytearray()
        self.client_bytes = self.client_frames = 0
        self.frames = 0
        self.bytes = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def observe(self, data):
        self.bytes += len(data)
        if self.bytes > getattr(self, "byte_limit", SERVER_BYTE_LIMIT):
            raise ValueError("wire byte limit exceeded")
        self.buffer.extend(data)
        for _ in range(16385):
            if len(self.buffer) < 4:
                return
            length = struct.unpack_from("<I", self.buffer)[0]
            if not 1 <= length <= FRAME_LIMIT:
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
            tick = verify_events(payload, getattr(self, "simulator_commit", CURRENT_SIMULATOR))
            if tick != self.observed.get("last_snapshot"):
                raise ValueError("Events snapshot tick mismatch")
            if tick == self.tick_limit:
                if self.observed.get("cap_events"):
                    raise ValueError("duplicate cap Events")
                self.observed["cap_events"] = True
        elif kind == 5:
            self.observed["rejected"] += 1
        elif kind == 7:
            self.observe_match_over(payload, offset)
        elif kind > 8:
            raise ValueError("unknown server message")

    def observe_match_over(self, payload, offset):
        cursor = Postcard(payload, offset)
        winner = cursor.integer()
        if winner not in (0, 1, 2) or self.observed["winner"] is not None:
            raise ValueError("invalid or duplicate MatchOver")
        duration, count = cursor.integer(), cursor.integer(64)
        if not 0 < duration <= self.tick_limit or count != 2:
            raise ValueError("invalid MatchOver duration or seat count")
        for index in range(count):
            if cursor.byte() != index:
                raise ValueError("MatchOver seat identity mismatch")
            for bits in (16, 16, 16, 16, 16, 32, 32, 32):
                cursor.integer(bits)
        if cursor.offset != len(payload):
            raise ValueError("trailing MatchOver bytes")
        self.observed["winner"] = ("Radiant", "Dire", "Neutral")[winner]
        self.observed["duration"] = duration

    def filter_client(self, data):
        self.client_bytes += len(data)
        if self.client_bytes > CLIENT_BYTE_LIMIT:
            raise ValueError("client wire byte limit exceeded")
        self.client_buffer.extend(data)
        output = bytearray()
        for _ in range(16385):
            if len(self.client_buffer) < 4:
                return output
            length = struct.unpack_from("<I", self.client_buffer)[0]
            if not 1 <= length <= FRAME_LIMIT:
                raise ValueError("invalid client frame length")
            if len(self.client_buffer) < length + 4:
                return output
            frame = bytes(self.client_buffer[:length + 4])
            del self.client_buffer[:length + 4]
            self.client_frames += 1
            if self.client_frames > self.tick_limit * 4 + 1000:
                raise ValueError("client wire frame limit exceeded")
            kind, tick = verify_client_message(frame[4:])
            if kind == 4:
                if tick > self.observed.get("last_snapshot", 0):
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
                    data = source.recv(READ_CHUNK)
                except ConnectionResetError:
                    if source is not client or not self.observed.get("cap_ack"):
                        raise
                    data = b""
                if not data:
                    if source is client:
                        if getattr(self, "client_buffer", None):
                            raise ValueError("truncated client frame")
                        self.observed["client_eof"] = True
                        if self.observed.get("cap_ack"):
                            readers.remove(client)
                            client_open = False
                            if not readers:
                                return
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
                    # A closed client read half can still send its pending tail.
                    if not readers:
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
                else:
                    # Terminal delivery disables forwarding, not client accounting.
                    filtered = self.filter_client(data)
                    try:
                        if filtered and writable and self.observed["winner"] is None:
                            server.sendall(filtered)
                    except (BrokenPipeError, ConnectionResetError):
                        # Drain the server's terminal frame before deciding whether EOF is valid.
                        writable = False
        raise ValueError("relay iteration limit exceeded")

    def finish(self, timeout=2):
        """Drain naturally after peer processes exit, before close; return whether observation is valid."""
        if not 0 <= timeout <= 2:
            raise ValueError("relay drain timeout must be 0..2 seconds")
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            self.observed["errors"].append("relay drain timeout")
        elif not self.observed.get("client_eof") and not self.observed["errors"]:
            self.observed["errors"].append("relay finished before client EOF")
        return not self.observed["errors"]

    def close(self):
        self.stop.set()
        self.listener.close()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            self.observed["errors"].append("relay failed to stop")
