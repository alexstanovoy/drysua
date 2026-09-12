"""Bounded, single-threaded Hello/Welcome admission for the pinned bota protocol."""

from dataclasses import dataclass, field
import errno
import select
import socket
import struct
import time

from release_wire import varint


HANDSHAKE_TIMEOUT = 30
PROGRESS_TIMEOUT = 30
SESSION_TIMEOUT = 4 * 60 * 60
FRAME_LIMIT = 4 * 1024 * 1024
QUEUE_LIMIT = FRAME_LIMIT + 4
READ_CHUNK = 65536
FRAME_COUNT_LIMIT = 1_000_000
SERVER_BYTE_LIMIT = 2 * 1024**3
CLIENT_BYTE_LIMIT = 64 * 1024**2
assert READ_CHUNK < FRAME_LIMIT < SERVER_BYTE_LIMIT
assert HANDSHAKE_TIMEOUT < SESSION_TIMEOUT


class Admission:
    """Two loopback relays; slot one cannot reach the server before Welcome zero."""

    def __init__(self, port, human_side, mode=0):
        assert 1 <= port <= 65535
        assert human_side in ("radiant", "dire")
        assert mode in (0, 1)
        self.relays = []
        self.deadline = time.monotonic() + SESSION_TIMEOUT
        roles = ("human", "bot") if human_side == "radiant" else ("bot", "human")
        try:
            for slot, role in enumerate(roles):
                self.relays.append(SeatRelay(port, slot, role, mode))
        except BaseException:
            self.close()
            raise
        self.addresses = {relay.role: relay.address for relay in self.relays}

    @property
    def welcomed(self):
        return all(relay.welcomed for relay in self.relays)

    def pump(self):
        assert len(self.relays) == 2
        if time.monotonic() >= self.deadline:
            raise RuntimeError(f"relay session exceeded {SESSION_TIMEOUT} seconds")
        progressed = False
        for relay in self.relays:
            relay.check_deadlines()
            # A queued TCP connection/Hello on listener one is not a server admission.
            if relay.slot == 0 or self.relays[0].welcomed:
                try:
                    progressed = relay.pump() or progressed
                except (OSError, ValueError) as error:
                    raise RuntimeError(f"{relay.role} relay: {error}") from error
        return progressed

    def close(self):
        assert len(self.relays) <= 2
        for relay in self.relays:
            relay.close()


def verify_hello(payload, role):
    assert role in ("human", "bot")
    expected_role, name = (0, b"human") if role == "human" else (1, b"drysua")
    # All fields are single-byte postcard integers for these fixed identities.
    expected = bytes([0, expected_role, len(name)]) + name
    if payload != expected:
        raise ValueError(f"invalid Hello: expected {role} identity and player/bot role")


def verify_welcome(payload, slot, mode):
    assert slot in (0, 1)
    assert mode in (0, 1)
    values, offset = [], 0
    try:
        for _ in range(6):
            value, offset = varint(payload, offset)
            values.append(value)
    except ValueError as error:
        raise ValueError(f"malformed Welcome: {error}") from error
    kind, player, present, actual, tick_rate, actual_mode = values
    if (kind != 0 or not 1 <= player <= 2**32 - 1 or present != 1
            or tick_rate != 30 or actual_mode != mode or offset != len(payload)):
        raise ValueError("invalid Welcome: expected seated participant, 30 Hz and requested tick mode")
    if actual != slot:
        raise ValueError(f"Welcome side mismatch: expected slot {slot}, got {actual}")


def verify_match_over(payload):
    statistics = (255,) + (65535,) * 5 + (2**32 - 1,) * 3
    limits = (7, 2, 2**32 - 1, 2) + statistics * 2
    values, offset = [], 0
    try:
        for maximum in limits:
            start = offset
            value, offset = varint(payload, offset)
            if value > maximum or offset - start != max(1, (value.bit_length() + 6) // 7):
                raise ValueError("out-of-range or noncanonical integer")
            values.append(value)
    except ValueError as error:
        raise ValueError(f"malformed MatchOver: {error}") from error
    if values[0] != 7 or values[2] == 0 or values[3] != 2 or offset != len(payload):
        raise ValueError("invalid MatchOver: expected duration, two seats and no trailing bytes")
    if values[4] != 0 or values[13] != 1:
        raise ValueError("invalid MatchOver seat identities")


@dataclass
class Endpoint:
    connection: socket.socket
    incoming: bytearray = field(default_factory=bytearray)
    outgoing: bytearray = field(default_factory=bytearray)
    eof: bool = False
    write_closed: bool = False
    connecting: bool = False
    received: int = 0
    frames: int = 0
    read_started: float = 0
    write_progress: float = 0

    def write(self):
        if self.connecting:
            error = self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if error:
                raise OSError(error, "connecting relay to loopback server")
            self.connecting = False
        if not self.outgoing:
            return
        try:
            count = self.connection.send(self.outgoing[:READ_CHUNK])
        except BlockingIOError:
            return
        if count == 0:
            raise OSError("relay socket made no write progress")
        del self.outgoing[:count]
        self.write_progress = time.monotonic()
        assert len(self.outgoing) <= QUEUE_LIMIT


class SeatRelay:
    def __init__(self, port, slot, role, mode):
        assert slot in (0, 1)
        assert role in ("human", "bot")
        self.port, self.slot, self.role, self.mode = port, slot, role, mode
        self.hello = self.welcomed = False
        self.match_over = False
        self.upstream_error = None
        self.upstream_deadline = 0
        self.endpoints = []
        self.deadline = time.monotonic() + HANDSHAKE_TIMEOUT
        self.listener = socket.socket()
        try:
            self.listener.bind(("127.0.0.1", 0))
            self.listener.listen(1)
            self.listener.setblocking(False)
            self.address = f"127.0.0.1:{self.listener.getsockname()[1]}"
        except BaseException:
            self.listener.close()
            raise

    def accept(self):
        client, _ = self.listener.accept()
        self.endpoints.append(Endpoint(client))
        self.listener.close()
        server = socket.socket()
        self.endpoints.append(Endpoint(server, connecting=True))
        for endpoint in self.endpoints:
            endpoint.connection.setblocking(False)
            endpoint.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        error = server.connect_ex(("127.0.0.1", self.port))
        if error not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK):
            raise OSError(error, "connecting relay to loopback server")
        assert len(self.endpoints) == 2

    def pump(self):
        if not self.endpoints:
            if select.select([self.listener], [], [], 0)[0]:
                self.accept()
                return True
            return False
        readers, writers = [], []
        for index, endpoint in enumerate(self.endpoints):
            target = self.endpoints[1 - index]
            if not endpoint.connecting and not endpoint.eof and self.read_budget(index):
                readers.append(endpoint.connection)
            if endpoint.connecting or (endpoint.outgoing and not endpoint.write_closed):
                writers.append(endpoint.connection)
            elif target.eof and not endpoint.write_closed:
                try:
                    endpoint.connection.shutdown(socket.SHUT_WR)
                except OSError as error:
                    if error.errno not in (errno.ENOTCONN, errno.EPIPE):
                        raise
                endpoint.write_closed = True
        readable, writable, _ = select.select(readers, writers, [], 0)
        # The final server frame can already be readable when a queued order hits EPIPE.
        for index in (1, 0):
            if self.endpoints[index].connection in readable:
                self.receive(index)
        for index, endpoint in enumerate(self.endpoints):
            if endpoint.connection in writable and not endpoint.write_closed:
                self.write_endpoint(index)
        client, server = self.endpoints
        if server.eof and not self.match_over and not client.eof:
            raise ValueError("server disconnected before verified MatchOver")
        return bool(readable or writable)

    def write_endpoint(self, index):
        endpoint = self.endpoints[index]
        try:
            endpoint.write()
        except (BrokenPipeError, ConnectionResetError) as error:
            if index != 1 or endpoint.connecting:
                raise
            # A failed upstream write is not success: drain and require a verified terminal frame.
            self.upstream_error = error
            self.upstream_deadline = time.monotonic() + PROGRESS_TIMEOUT
            endpoint.write_closed = True
            endpoint.outgoing.clear()

    def read_budget(self, index):
        source, target = self.endpoints[index], self.endpoints[1 - index]
        available = QUEUE_LIMIT - len(source.incoming) - len(target.outgoing)
        assert 0 <= available <= QUEUE_LIMIT
        return min(READ_CHUNK, available)

    def receive(self, index):
        source, target = self.endpoints[index], self.endpoints[1 - index]
        try:
            data = source.connection.recv(self.read_budget(index))
        except BlockingIOError:
            return
        except ConnectionResetError as error:
            if source.incoming:
                raise ValueError("truncated relay frame at reset") from error
            if not self.match_over:
                raise ValueError("connection reset before verified MatchOver") from error
            if index == 0 and source.outgoing:
                raise ValueError("client reset before final-frame drain") from error
            data = b""
        if not data:
            if source.incoming:
                raise ValueError("truncated relay frame at EOF")
            source.eof = True
            return
        source.received += len(data)
        limit = SERVER_BYTE_LIMIT if index else CLIENT_BYTE_LIMIT
        if source.received > limit:
            raise ValueError(f"relay wire byte limit exceeded ({limit} bytes)")
        if not source.incoming:
            source.read_started = time.monotonic()
        source.incoming.extend(data)
        self.forward_frames(index)
        assert len(source.incoming) + len(target.outgoing) <= QUEUE_LIMIT

    def forward_frames(self, index):
        source, target = self.endpoints[index], self.endpoints[1 - index]
        buffer, offset = source.incoming, 0
        for _ in range(READ_CHUNK // 5 + 2):
            if len(buffer) - offset < 4:
                break
            length = struct.unpack_from("<I", buffer, offset)[0]
            first = not (self.welcomed if index else self.hello)
            limit = 64 if first else FRAME_LIMIT
            if not 1 <= length <= limit:
                raise ValueError(f"invalid relay frame length {length}; limit {limit}")
            end = offset + 4 + length
            if len(buffer) < end:
                break
            source.frames += 1
            if source.frames > FRAME_COUNT_LIMIT:
                raise ValueError("relay frame count limit exceeded")
            self.observe(bytes(buffer[offset + 4:end]), index)
            if index == 1 or not (self.match_over or self.upstream_error):
                if not target.outgoing:
                    target.write_progress = time.monotonic()
                target.outgoing.extend(buffer[offset:end])
            offset = end
        else:
            raise ValueError("relay frame batch limit exceeded")
        if offset:
            del buffer[:offset]
            source.read_started = time.monotonic()

    def observe(self, payload, index):
        if index == 0 and not self.hello:
            verify_hello(payload, self.role)
            self.hello = True
        elif index == 1 and not self.welcomed:
            # Bota broadcasts lobby changes to accepted sockets that have not said Hello yet.
            if payload[:1] == b"\x01":
                return
            if not self.hello:
                raise ValueError("Welcome arrived before Hello")
            verify_welcome(payload, self.slot, self.mode)
            self.welcomed = True
            side = ("Radiant", "Dire")[self.slot]
            print(f"play: verified {self.role} {side}: Welcome slot {self.slot}", flush=True)
        else:
            if index == 1 and self.match_over:
                raise ValueError("unexpected server message after MatchOver")
            kind, offset = varint(payload, 0)
            if kind == 0 or kind > (8 if index else 5):
                raise ValueError("duplicate handshake or unknown relay message")
            if index == 1 and kind == 3:
                _, offset = varint(payload, offset)
                present, offset = varint(payload, offset)
                team, _ = varint(payload, offset)
                if present != 1 or team != self.slot:
                    raise ValueError("snapshot side differs from verified Welcome")
            if index == 1 and kind == 7:
                verify_match_over(payload)
                self.match_over = True
                self.endpoints[1].outgoing.clear()
                self.endpoints[1].write_closed = True

    def check_deadlines(self):
        now = time.monotonic()
        if not self.welcomed and now >= self.deadline:
            raise RuntimeError(f"{self.role} Hello/Welcome timed out after {HANDSHAKE_TIMEOUT} seconds")
        if self.upstream_error and not self.match_over and now >= self.upstream_deadline:
            raise RuntimeError(f"{self.role} server write failed before verified MatchOver "
                               f"({PROGRESS_TIMEOUT}-second drain deadline): {self.upstream_error}")
        for endpoint in self.endpoints:
            if endpoint.incoming and now - endpoint.read_started >= PROGRESS_TIMEOUT:
                raise RuntimeError(f"{self.role} relay incomplete frame timed out")
            if endpoint.outgoing and now - endpoint.write_progress >= PROGRESS_TIMEOUT:
                raise RuntimeError(f"{self.role} relay blocked write timed out")

    def close(self):
        assert len(self.endpoints) <= 2
        self.listener.close()
        for endpoint in self.endpoints:
            endpoint.connection.close()
