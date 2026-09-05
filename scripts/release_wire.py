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

    def __init__(self, port, timeout, tick_limit):
        self.port = port
        self.timeout = timeout
        self.tick_limit = tick_limit
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(timeout)
        self.address = f"127.0.0.1:{self.listener.getsockname()[1]}"
        self.welcomed = threading.Event()
        self.stop = threading.Event()
        self.observed = dict(slot=None, winner=None, rejected=0, errors=[])
        self.buffer = bytearray()
        self.frames = 0
        self.bytes = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def observe(self, data):
        self.bytes += len(data)
        if self.bytes > 512 * 1024 * 1024:
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
        elif kind == 3:
            tick, offset = varint(payload, offset)
            if tick > self.tick_limit:
                raise ValueError("server exceeded tick limit")
            present, offset = varint(payload, offset)
            viewer, _ = varint(payload, offset)
            if present != 1 or viewer != self.observed["slot"]:
                raise ValueError("snapshot identity mismatch")
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
        for _ in range(self.tick_limit * 40 + 4000):
            if self.stop.is_set():
                return
            readable, _, _ = select.select(readers, [], [], 0.1)
            for source in readable:
                data = source.recv(65536)
                if not data:
                    if source is client:
                        return
                    if self.buffer:
                        raise ValueError("truncated server frame")
                    if self.observed["winner"] is None:
                        raise ValueError("server closed before MatchOver")
                    # Closing with unread final ACKs sends RST instead of delivering MatchOver.
                    readers.remove(server)
                    writable = False
                    client.shutdown(socket.SHUT_WR)
                elif source is server:
                    self.observe(data)
                    client.sendall(data)
                elif writable:
                    try:
                        server.sendall(data)
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
