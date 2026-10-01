//! Replay depends on the action log codec: every action must survive a round
//! trip, and any word the encoder cannot produce must be rejected.

use super::*;
use bota_proto::{AbilitySlot, ItemSlot};

fn actions() -> impl Iterator<Item = StructuredAction> {
    body_actions().into_iter().chain(item_actions())
}

fn body_actions() -> [StructuredAction; 9] {
    use StructuredAction as Action;
    let unit = ControlledUnit::Courier;
    [
        Action::Continue,
        Action::Stop { unit },
        Action::MovePoint {
            unit,
            point: PointIndex(47),
        },
        Action::FollowUnit {
            unit,
            target: EntityIndex(95),
        },
        Action::Hold {
            unit: ControlledUnit::Hero,
        },
        Action::AttackMovePoint {
            unit,
            point: PointIndex(0),
        },
        Action::AttackUnit {
            unit,
            target: EntityIndex(3),
        },
        Action::Cast {
            unit,
            slot: AbilitySlot(7),
            target: ActionTarget::None,
        },
        Action::Cast {
            unit,
            slot: AbilitySlot(1),
            target: ActionTarget::Entity(EntityIndex(95)),
        },
    ]
}

fn item_actions() -> [StructuredAction; 9] {
    use StructuredAction as Action;
    let unit = ControlledUnit::Courier;
    [
        Action::Use {
            unit,
            slot: ItemSlot(14),
            target: ActionTarget::Point(PointIndex(47)),
        },
        Action::PutPoint {
            unit,
            source: ItemSlot(2),
            target: PutPointTarget::Underfoot,
        },
        Action::PutPoint {
            unit,
            source: ItemSlot(2),
            target: PutPointTarget::Point(PointIndex(9)),
        },
        Action::PutUnit {
            unit,
            source: ItemSlot(5),
            target: EntityIndex(0),
        },
        Action::Take {
            unit,
            loot: LootIndex(15),
        },
        Action::Buy {
            unit,
            item: ShopIndex(63),
        },
        Action::Sell {
            unit,
            slot: ItemSlot(8),
        },
        Action::Swap {
            unit,
            from: ItemSlot(0),
            to: ItemSlot(14),
        },
        Action::Learn {
            slot: AbilitySlot(5),
        },
    ]
}

#[test]
fn every_action_family_round_trips_through_the_log_codec() {
    for action in actions() {
        let word = encode_action(action).expect("encodable action");
        assert_eq!(decode_action(word).expect("decodable word"), action);
    }
}

#[test]
fn log_codec_rejects_words_the_encoder_never_writes() {
    let continue_word = encode_action(StructuredAction::Continue).expect("continue");
    let put_point = encode_action(StructuredAction::PutPoint {
        unit: ControlledUnit::Hero,
        source: ItemSlot(1),
        target: PutPointTarget::Underfoot,
    })
    .expect("put point");
    for corrupt in [
        continue_word | 1 << 4,
        continue_word | 1 << 5,
        continue_word | 1 << 23,
        continue_word | 1 << 31,
        put_point | 1 << 21,
        encode_action(StructuredAction::Hold {
            unit: ControlledUnit::Hero,
        })
        .expect("hold")
            | 3 << 21,
    ] {
        assert_eq!(
            decode_action(corrupt),
            Err(PpoError::InvalidConfig("action log word")),
            "{corrupt:#x}"
        );
    }
}

#[test]
fn log_codec_rejects_an_index_beyond_one_byte() {
    let error = encode_action(StructuredAction::MovePoint {
        unit: ControlledUnit::Hero,
        point: PointIndex(256),
    });
    assert_eq!(error, Err(PpoError::InvalidTransition("action log index")));
}
