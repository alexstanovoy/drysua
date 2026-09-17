"""Pinned postcard event/order vectors; no processes, sockets, clocks, or matches.

Every fixture is bound to exactly one simulator pin. Current (78427bb) inserted
Missed after Damaged and shifted the later variants; the previous (037c6a2) and
historical (18db0f6) pins keep their own order and are never decoded as current.
"""

import struct
import unittest

import release_wire as wire


CURRENT = wire.CURRENT_SIMULATOR
PREVIOUS = wire.PREVIOUS_SIMULATOR
HISTORICAL = wire.HISTORICAL_SIMULATOR
HEAL_TAG = {CURRENT: b"\x02", PREVIOUS: b"\x01", HISTORICAL: b"\x01"}
# Previous-pin event bodies in the pre-Missed declaration order.
HP_ONLY_OLD = bytes.fromhex("01 01 ac02 02 ad02 03 fa01 00")
MANA_ONLY_OLD = bytes.fromhex("01 00 ad02 03 00 ac02")
BOTH_OLD = bytes.fromhex("01 01 ac02 02 ad02 03 5a f001")
# Current order: Damage(0), Missed(1), Healed(2), Died(3), Cast(4), Level(5),
# purchase(6), structure(7). Missed carries only an optional source and target.
FOLLOWING_CURRENT = bytes.fromhex(
    "00 01 ac02 02 ad02 03 9003 01 01 "
    "01 01 ac02 02 ad02 03 "
    "02 01 ac02 02 ad02 03 5a f001 "
    "03 ad02 03 01 ac02 02 00 a006 "
    "04 ac02 02 c801 "
    "05 ac02 02 ff "
    "06 01 c901 "
    "07 ae02 04 01")

# Damage, death, cast, level, purchase, structure under the previous pin.
FOLLOWING_OLD = bytes.fromhex(
    "00 01 ac02 02 ad02 03 9003 01 01 "
    "02 ad02 03 01 ac02 02 00 a006 "
    "03 ac02 02 c801 04 ac02 02 ff 05 01 c901 06 ae02 04 01")


def rebase_heal(fixture, simulator):
    """Re-tag a previous-pin heal body for another pin; Missed shifts only tags."""
    if simulator == PREVIOUS or simulator == HISTORICAL:
        return fixture
    return HEAL_TAG[simulator] + fixture[1:]


def rebase_bytes(fixture, simulator):
    """Re-tag a previous-pin mixed batch for the current shifted order."""
    if simulator != CURRENT:
        return fixture
    return fixture.replace(b"\x01\x01 ac02", b"\x02\x01 ac02", 1).replace(
        b"\x01\x00 ad02", b"\x02\x00 ad02", 1).replace(
        b"\x01\x01 ac02 02 ad02 03 5a f001", b"\x02\x01 ac02 02 ad02 03 5a f001", 1)


# Convenience current-pin heal batch for helpers that default to the current pin.
BOTH_CURRENT = rebase_heal(BOTH_OLD, CURRENT)


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


def order_message(order_body):
    # ClientMsg::Order: kind, sequence, optional entity (flag and two varints).
    return b"\x03" + integer(7) + b"\x01" + integer(300) + integer(2) + order_body


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
        2 * 325 * 65536, 2 * 20, 2 * 100 * 65536, 30, 30, 2 * 100,
        2 * 2 * 65536, 2 * 16384, 2 * 27 * 65536, 2 * 24 * 65536,
        2 * 800 * 65536, 0, 0,           # Vision, true sight, statuses.
        0, 0, 0, 0, 0, 0, 0,            # Attributes, primary, hero, owner, level.
        0, 0, 3,                        # Abilities, items, effects lengths.
    )
    effects = bytes.fromhex("0d 01 8407 00 0e 00 00 0f 01 9601 01 02")
    unit = b"".join(integer(value) for value in fields) + effects
    return b"\x03" + integer(901) + b"\x01\x00\x01" + unit + bytes(5)


class CurrentEventTests(unittest.TestCase):
    def test_current_missed_is_accepted_and_consumed_before_later_variants(self):
        payload = events(FOLLOWING_CURRENT, 8)
        self.assertEqual(wire.verify_events(payload), 901)
        relay = observer()
        relay.observe(frame(payload))
        self.assertTrue(relay.observed["cap_events"])
        self.assertEqual(relay.buffer, b"")

    def test_current_heals_shift_to_index_two_and_consume_mana(self):
        for healed in (HP_ONLY_OLD, MANA_ONLY_OLD, BOTH_OLD):
            with self.subTest(healed=healed.hex()):
                self.assertEqual(wire.verify_events(events(rebase_heal(healed, CURRENT))), 901)

    def test_current_shifted_following_variants_stay_aligned(self):
        payload = events(FOLLOWING_CURRENT, 8)
        self.assertEqual(wire.verify_events(payload), 901)
        relay = observer()
        relay.observe(frame(payload))
        self.assertTrue(relay.observed["cap_events"])

    def test_current_invalid_kinds_and_fields_fail_closed(self):
        invalid = (
            (events(b"\x08"), "unknown event kind"),
            (events(b"\x01\x02"), "invalid postcard option"),
            (events(b"\x02\x00" + integer(2**32)), "postcard integer exceeds u32"),
            (events(bytes.fromhex("06 00") + integer(2**16)), "postcard integer exceeds u16"),
            (events(bytes.fromhex("07 01 00 03")), "invalid event team"),
            (events(bytes.fromhex("00 00 01 00 02 03 00")), "invalid damage kind"),
            (events(bytes.fromhex("00 00 01 00 02 00 02")), "invalid postcard bool"),
        )
        for payload, message in invalid:
            with self.subTest(payload=payload.hex()), self.assertRaisesRegex(ValueError, message):
                wire.verify_events(payload)

    def test_missed_has_no_amount_kind_or_crit_fields(self):
        # One byte too few for Damaged framing, exactly right for Missed.
        body = bytes.fromhex("00 01 ac02 02 ad02 03 9003 01")
        with self.assertRaisesRegex(ValueError, "truncated"):
            wire.verify_events(events(body))


class PreviousPinTests(unittest.TestCase):
    def test_previous_heals_keep_the_old_declaration_order(self):
        for healed in (HP_ONLY_OLD, MANA_ONLY_OLD, BOTH_OLD):
            with self.subTest(healed=healed.hex()):
                payload = events(healed + FOLLOWING_OLD, 7)
                self.assertEqual(wire.verify_events(payload, PREVIOUS), 901)
                relay = observer()
                relay.simulator_commit = PREVIOUS
                relay.observe(frame(payload))
                self.assertTrue(relay.observed["cap_events"])
                self.assertEqual(relay.buffer, b"")

    def test_current_missed_tag_is_healed_under_the_previous_pin(self):
        # A current-shaped Missed first byte cannot be guessed as a miss.
        with self.assertRaisesRegex(ValueError, "truncated|invalid"):
            wire.verify_events(events(bytes.fromhex("01 01 ac02 02 ad02")), PREVIOUS)

    def test_previous_truncations_and_trailing_bytes_fail(self):
        payload = events(BOTH_OLD + FOLLOWING_OLD, 7)
        for length in range(len(payload)):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "truncated"):
                wire.verify_events(payload[:length], PREVIOUS)
        with self.assertRaisesRegex(ValueError, "trailing Events bytes"):
            wire.verify_events(events(BOTH_OLD) + b"\x00", PREVIOUS)

    def test_historical_heal_requires_explicit_pin_and_is_not_guessed(self):
        old_payload = events(HP_ONLY_OLD[:-1])
        self.assertEqual(wire.verify_events(old_payload, HISTORICAL), 901)
        with self.assertRaisesRegex(ValueError, "truncated postcard integer"):
            wire.verify_events(old_payload, PREVIOUS)
        with self.assertRaisesRegex(ValueError, "trailing Events bytes"):
            wire.verify_events(events(HP_ONLY_OLD), HISTORICAL)
        with self.assertRaisesRegex(ValueError, "unsupported simulator wire contract"):
            wire.verify_events(events(BOTH_OLD), "unknown")


class ClientOrderTests(unittest.TestCase):
    def test_current_cheat_variants_are_decoded_structurally(self):
        variants = (
            bytes([10, 0]) + integer(2 * 100 + 1),  # Gold +100 zigzag.
            bytes([10, 1, 3]),                      # Levels 3.
            bytes([10, 2]),                         # Refresh.
            bytes([10, 3]) + integer(46),           # Item 46.
        )
        for body in variants:
            with self.subTest(body=body.hex()):
                kind, tick = wire.verify_client_message(order_message(body), CURRENT)
                self.assertEqual((kind, tick), (3, None))
        self.assertEqual(wire.verify_client_message(order_message(bytes([10, 0]) + integer(1)), CURRENT)[0], 3)

    def test_cheat_is_unknown_under_previous_and_historical_pins(self):
        body = order_message(bytes([10, 2]))
        for simulator in (PREVIOUS, HISTORICAL):
            with self.subTest(simulator=simulator), self.assertRaisesRegex(ValueError, "unknown client order"):
                wire.verify_client_message(body, simulator)

    def test_unknown_cheat_variant_and_truncated_cheat_fail(self):
        for body, message in ((bytes([10, 4]), "unknown cheat"),
                              (bytes([10]), "truncated postcard integer"),
                              (bytes([10, 3]), "truncated postcard integer")):
            with self.subTest(body=body.hex()), self.assertRaisesRegex(ValueError, message):
                wire.verify_client_message(order_message(body), CURRENT)

    def test_ordinary_orders_keep_their_previous_encoding(self):
        bodies = (
            bytes([0, 2]) + integer(1000) + integer(2000),  # Move point.
            bytes([1, 0]),                                   # Attack nothing.
            bytes([2, 1]) + integer(0),                      # Cast slot 1 on nothing.
            bytes([3, 0]) + integer(0),                      # Use slot 0 on nothing.
            bytes([4, 0, 0]),                                # Put nothing.
            bytes([5, 0]),                                   # Take nothing.
            bytes([6]) + integer(46),                        # Buy 46.
            bytes([7, 0]),                                   # Sell slot 0.
            bytes([8, 0, 1]),                                # Swap.
            bytes([9, 2]),                                   # Learn slot 2.
        )
        for body in bodies:
            with self.subTest(body=body.hex()):
                for simulator in (CURRENT, PREVIOUS, HISTORICAL):
                    self.assertEqual(wire.verify_client_message(order_message(body), simulator)[0], 3)


class FramingTests(unittest.TestCase):
    def test_fragmented_current_events_then_native_draw_preserve_frame_alignment(self):
        relay = observer(27900)
        over = b"\x07\x02" + integer(27900) + b"\x02" + bytes(9) + b"\x01" + bytes(8)
        heal = rebase_heal(BOTH_OLD, CURRENT)
        data = frame(events(heal * 3, 3, 27900)) + frame(over)
        for byte in data:
            relay.observe(bytes([byte]))
        self.assertEqual(relay.observed["winner"], "Neutral")
        self.assertEqual(relay.observed["duration"], 27900)
        self.assertEqual(relay.frames, 2)

    def test_truncated_queued_events_are_not_a_verified_cap(self):
        relay = observer()
        relay.observe(frame(events(rebase_heal(MANA_ONLY_OLD, CURRENT)))[:-1])
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

    def test_duplicate_heals_and_empty_batch_are_valid_under_current(self):
        for payload in (events(rebase_heal(BOTH_OLD, CURRENT) * 3, 3), events(b"", 0)):
            self.assertEqual(wire.verify_events(payload), 901)


if __name__ == "__main__":
    unittest.main()
