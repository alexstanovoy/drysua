"""Current postcard event vectors; no processes, sockets, clocks, or matches."""

import struct
import unittest

import release_wire as wire


CURRENT = "037c6a2f8e5383beae9eea6da8cbbb1678f7b718"
HISTORICAL = "18db0f62d9a2b94e755c43fd29a959db204cc20b"
HP_ONLY = bytes.fromhex("01 01 ac02 02 ad02 03 fa01 00")
MANA_ONLY = bytes.fromhex("01 00 ad02 03 00 ac02")
BOTH = bytes.fromhex("01 01 ac02 02 ad02 03 5a f001")
# Damage, death, cast, level, purchase, structure, in EventKind declaration order.
FOLLOWING = bytes.fromhex(
    "00 01 ac02 02 ad02 03 9003 01 01 "
    "02 ad02 03 01 ac02 02 00 a006 "
    "03 ac02 02 c801 04 ac02 02 ff 05 01 c901 06 ae02 04 01")


def integer(value):
    assert 0 <= value < 2**64
    data = bytearray()
    for _ in range(10):
        data.append((value & 127) | (128 if value >= 128 else 0))
        value >>= 7
        if not value:
            return bytes(data)
    raise AssertionError("fixture integer exceeds u64")


def events(body, count=1, tick=901):
    return b"\x04" + integer(tick) + integer(count) + body


def frame(payload):
    assert 0 < len(payload) <= 4 * 1024 * 1024
    return struct.pack("<I", len(payload)) + payload


def observer(tick=901):
    relay = wire.Relay.__new__(wire.Relay)
    relay.tick_limit = tick
    relay.simulator_commit = CURRENT
    relay.buffer = bytearray()
    relay.client_buffer = bytearray()
    relay.bytes = relay.frames = relay.client_bytes = relay.client_frames = 0
    relay.observed = dict(slot=0, winner=None, rejected=0, errors=[], last_snapshot=tick,
                          cap_events=False, cap_ack=False, map=2)
    return relay


def snapshot_with_effects():
    # Current UnitView: a visible melee creep; Fixed and i32 fields use zigzag u32.
    fields = (
        300, 2, 1, 0,                     # Entity, UnitKind, Team.
        2 * 100 * 65536, 2 * 100 * 65536, 0,  # Position and facing.
        2 * 550, 2 * 550, 0, 0,          # Health and mana.
        2 * 325 * 65536, 2 * 20, 2 * 100 * 65536, 30, 2 * 100,
        2 * 2 * 65536, 2 * 16384, 2 * 16 * 65536,
        2 * 800 * 65536, 0, 0,           # Vision, true sight, statuses.
        0, 0, 0, 0, 0, 0, 0,            # Attributes, primary, hero, owner, level.
        0, 0, 3,                        # Abilities, items, effects lengths.
    )
    effects = bytes.fromhex("0d 01 8407 00 0e 00 00 0f 01 9601 01 02")
    unit = b"".join(integer(value) for value in fields) + effects
    # WorldView: tick/viewer, units, projectiles, players, trees, planted trees, loot.
    return b"\x03" + integer(901) + b"\x01\x00\x01" + unit + bytes(5)


class EventTests(unittest.TestCase):
    def test_generic_effect_ids_13_14_15_do_not_change_snapshot_or_next_event_framing(self):
        relay = observer()
        relay.observed["last_snapshot"] = 900
        data = frame(snapshot_with_effects()) + frame(events(BOTH))
        relay.observe(data)
        self.assertEqual(relay.observed["last_snapshot"], 901)
        self.assertTrue(relay.observed["cap_events"])
        self.assertEqual(relay.frames, 2)
        self.assertEqual(relay.buffer, b"")

    def test_current_heals_consume_health_and_mana_before_every_following_variant(self):
        for healed in (HP_ONLY, MANA_ONLY, BOTH):
            with self.subTest(healed=healed.hex()):
                payload = events(healed + FOLLOWING, 7)
                self.assertEqual(wire.verify_events(payload), 901)
                relay = observer()
                relay.observe(frame(payload))
                self.assertTrue(relay.observed["cap_events"])
                self.assertEqual(relay.buffer, b"")

    def test_duplicate_individual_heals_and_empty_batch_are_valid(self):
        for payload in (events(BOTH * 3, 3), events(b"", 0)):
            self.assertEqual(wire.verify_events(payload), 901)

    def test_missing_truncated_overflow_and_noncanonical_mana_cannot_verify_cap(self):
        prefix = HP_ONLY[:-1]
        for mana, message in ((b"", "truncated postcard integer"),
                              (b"\x80", "truncated postcard integer"),
                              (integer(2**32), "postcard integer exceeds u32"),
                              (b"\x80\x00", "noncanonical postcard integer")):
            with self.subTest(mana=mana.hex()):
                relay = observer()
                with self.assertRaisesRegex(ValueError, message):
                    relay.observe(frame(events(prefix + mana)))
                self.assertFalse(relay.observed["cap_events"])

    def test_signed_i32_wire_boundaries_remain_representable(self):
        # Signed values are zigzag u32, not semantic HP/mana validation.
        prefix = bytes.fromhex("01 00 ad02 03")
        self.assertEqual(wire.verify_events(events(prefix + integer(2**32 - 1) * 2)), 901)

    def test_every_truncation_of_a_heal_and_following_batch_fails(self):
        payload = events(BOTH + FOLLOWING, 7)
        for length in range(len(payload)):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "truncated"):
                wire.verify_events(payload[:length])

    def test_invalid_event_fields_and_trailing_bytes_fail_closed(self):
        invalid = (
            (events(b"\x07"), "unknown event kind"),
            (events(b"\x01\x02"), "invalid postcard option"),
            (events(b"\x01\x00" + integer(2**32)), "postcard integer exceeds u32"),
            (events(bytes.fromhex("05 00") + integer(2**16)), "postcard integer exceeds u16"),
            (events(bytes.fromhex("06 01 00 03")), "invalid event team"),
            (events(bytes.fromhex("00 00 01 00 02 03 00")), "invalid damage kind"),
            (events(bytes.fromhex("00 00 01 00 02 00 02")), "invalid postcard bool"),
            (events(BOTH) + b"\x00", "trailing Events bytes"),
            (events(b"", 4 * 1024 * 1024 // 3 + 1), "event count limit exceeded"),
            (events(b"", tick=2**32), "postcard integer exceeds u32"),
        )
        for payload, message in invalid:
            with self.subTest(payload=payload.hex()), self.assertRaisesRegex(ValueError, message):
                wire.verify_events(payload)

    def test_historical_heal_requires_explicit_pin_and_is_not_guessed(self):
        old_payload = events(HP_ONLY[:-1])
        self.assertEqual(wire.verify_events(old_payload, HISTORICAL), 901)
        with self.assertRaisesRegex(ValueError, "truncated postcard integer"):
            wire.verify_events(old_payload)
        with self.assertRaisesRegex(ValueError, "trailing Events bytes"):
            wire.verify_events(events(HP_ONLY), HISTORICAL)
        with self.assertRaisesRegex(ValueError, "unsupported simulator wire contract"):
            wire.verify_events(events(BOTH), "unknown")

    def test_fragmented_events_then_native_draw_preserve_frame_alignment(self):
        relay = observer(27900)
        over = b"\x07\x02" + integer(27900) + b"\x02" + bytes(9) + b"\x01" + bytes(8)
        data = frame(events(HP_ONLY + MANA_ONLY + BOTH, 3, 27900)) + frame(over)
        for byte in data:
            relay.observe(bytes([byte]))
        self.assertEqual(relay.observed["winner"], "Neutral")
        self.assertEqual(relay.observed["duration"], 27900)
        self.assertEqual(relay.frames, 2)

    def test_truncated_queued_events_are_not_a_verified_cap(self):
        relay = observer()
        relay.observe(frame(events(MANA_ONLY))[:-1])
        self.assertFalse(relay.observed["cap_events"])
        self.assertTrue(relay.buffer)


class BoundTests(unittest.TestCase):
    def test_varint_u64_maximum_is_valid_but_overflow_and_negative_offset_fail(self):
        self.assertEqual(wire.varint(integer(2**64 - 1), 0), (2**64 - 1, 10))
        for payload, offset, message in ((b"\xff" * 9 + b"\x02", 0, "exceeds u64"),
                                         (b"\x00", -1, "invalid postcard offset"),
                                         (b"\x80\x00", 0, "noncanonical postcard integer")):
            with self.assertRaisesRegex(ValueError, message):
                wire.varint(payload, offset)

    def test_frame_bytes_and_count_limits_remain_enforced(self):
        relay = observer()
        relay.byte_limit = len(frame(events(b"", 0)))
        relay.observe(frame(events(b"", 0)))
        with self.assertRaisesRegex(ValueError, "wire byte limit exceeded"):
            relay.observe(b"\x00")
        for size in (0, 4 * 1024 * 1024 + 1):
            with self.assertRaisesRegex(ValueError, "invalid wire frame length"):
                observer().observe(struct.pack("<I", size))
        relay = observer()
        relay.frames = relay.tick_limit * 10 + 1000
        with self.assertRaisesRegex(ValueError, "wire frame limit exceeded"):
            relay.observe(frame(events(b"", 0)))

    def test_client_byte_and_frame_bounds_and_direct_event_payload_bound_remain_enforced(self):
        relay = observer()
        relay.client_bytes = 64 * 1024 * 1024
        with self.assertRaisesRegex(ValueError, "client wire byte limit exceeded"):
            relay.filter_client(b"\x00")
        for size in (0, 4 * 1024 * 1024 + 1):
            with self.assertRaisesRegex(ValueError, "invalid client frame length"):
                observer().filter_client(struct.pack("<I", size))
        with self.assertRaisesRegex(ValueError, "wire payload limit exceeded"):
            wire.verify_events(bytes(4 * 1024 * 1024 + 1))

    def test_matchover_stat_overflow_does_not_publish_winner(self):
        relay = observer()
        over = b"\x07\x02" + integer(901) + b"\x02\x00" + integer(2**16) + bytes(7)
        over += b"\x01" + bytes(8)
        with self.assertRaisesRegex(ValueError, "postcard integer exceeds u16"):
            relay.observe_message(over)
        self.assertIsNone(relay.observed["winner"])


if __name__ == "__main__":
    unittest.main()
