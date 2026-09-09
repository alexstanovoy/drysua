use super::*;
use bota_proto::{EntityId, Fixed, ItemId, ItemSlot, Target, Vec2};

pub(super) fn parse_request(text: &str) -> Result<Option<Request>, &'static str> {
    if text.len() > 2048 {
        return Err("replay request exceeds bound");
    }
    if text == "None" {
        return Ok(None);
    }
    let inner = text
        .strip_prefix("Some(Request { seq: ")
        .and_then(|value| value.strip_suffix(" })"))
        .ok_or("replay request envelope")?;
    let (sequence, rest) = inner
        .split_once(", unit: ")
        .ok_or("replay request sequence")?;
    let (unit, order) = rest
        .split_once(", order: ")
        .ok_or("replay request fields")?;
    let request = Some(Request {
        seq: sequence.parse().map_err(|_| "replay sequence integer")?,
        unit: if unit == "None" {
            None
        } else {
            Some(parse_entity(
                unit.strip_prefix("Some(")
                    .and_then(|value| value.strip_suffix(')'))
                    .ok_or("replay controlled body")?,
            )?)
        },
        order: parse_order(order)?,
    });
    if format!("{request:?}") != text {
        return Err("replay request is not canonical or lossless");
    }
    Ok(request)
}

fn parse_order(text: &str) -> Result<Order, &'static str> {
    let (kind, fields) = text.split_once(" { ").ok_or("replay order fields")?;
    let fields = fields.strip_suffix(" }").ok_or("replay order closing")?;
    let target = || {
        parse_target(
            fields
                .strip_prefix("target: ")
                .ok_or("replay target field")?,
        )
    };
    let slot = || fields.strip_prefix("slot: ").ok_or("replay slot field");
    match kind {
        "Move" => Ok(Order::Move { target: target()? }),
        "Attack" => Ok(Order::Attack { target: target()? }),
        "Take" => Ok(Order::Take { target: target()? }),
        "Buy" => Ok(Order::Buy {
            item: ItemId(
                wrapped_number(
                    fields.strip_prefix("item: ").ok_or("replay item field")?,
                    "ItemId",
                )?
                .try_into()
                .map_err(|_| "replay item range")?,
            ),
        }),
        "Learn" => Ok(Order::Learn {
            slot: AbilitySlot(
                wrapped_number(slot()?, "AbilitySlot")?
                    .try_into()
                    .map_err(|_| "replay ability range")?,
            ),
        }),
        "Sell" => Ok(Order::Sell {
            slot: item_slot(slot()?)?,
        }),
        "Swap" => {
            let (source, target) = fields
                .strip_prefix("from: ")
                .and_then(|value| value.split_once(", to: "))
                .ok_or("replay swap fields")?;
            Ok(Order::Swap {
                from: item_slot(source)?,
                to: item_slot(target)?,
            })
        }
        "Cast" | "Use" | "Put" => parse_slotted_order(kind, slot()?),
        _ => Err("unsupported replay order kind"),
    }
}

fn parse_slotted_order(kind: &str, fields: &str) -> Result<Order, &'static str> {
    assert!(["Cast", "Use", "Put"].contains(&kind));
    assert!(fields.len() <= 2048);
    let (slot, target) = fields
        .split_once(", target: ")
        .ok_or("replay slot target fields")?;
    let target = parse_target(target)?;
    match kind {
        "Cast" => Ok(Order::Cast {
            slot: AbilitySlot(
                wrapped_number(slot, "AbilitySlot")?
                    .try_into()
                    .map_err(|_| "replay ability range")?,
            ),
            target,
        }),
        "Use" => Ok(Order::Use {
            slot: item_slot(slot)?,
            target,
        }),
        "Put" => Ok(Order::Put {
            slot: item_slot(slot)?,
            target,
        }),
        _ => unreachable!("bounded order alternatives"),
    }
}

fn parse_target(text: &str) -> Result<Target, &'static str> {
    if text == "None" {
        return Ok(Target::None);
    }
    if let Some(entity) = text
        .strip_prefix("Unit(")
        .and_then(|value| value.strip_suffix(')'))
    {
        return Ok(Target::Unit(parse_entity(entity)?));
    }
    let inner = text
        .strip_prefix("Pos(Vec2 { x: ")
        .and_then(|value| value.strip_suffix(" })"))
        .ok_or("replay target variant")?;
    let (x, y) = inner.split_once(", y: ").ok_or("replay point fields")?;
    Ok(Target::Pos(Vec2 {
        x: parse_fixed(x)?,
        y: parse_fixed(y)?,
    }))
}

fn parse_entity(text: &str) -> Result<EntityId, &'static str> {
    let inner = text
        .strip_prefix("EntityId { idx: ")
        .and_then(|value| value.strip_suffix(" }"))
        .ok_or("replay entity fields")?;
    let (index, generation) = inner
        .split_once(", generation: ")
        .ok_or("replay entity generation")?;
    Ok(EntityId {
        idx: index.parse().map_err(|_| "replay entity index integer")?,
        generation: generation
            .parse()
            .map_err(|_| "replay entity generation integer")?,
    })
}

fn parse_fixed(text: &str) -> Result<Fixed, &'static str> {
    if text.len() > 16 {
        return Err("replay fixed length");
    }
    let (negative, magnitude) = text
        .strip_prefix('-')
        .map_or((false, text), |value| (true, value));
    let (whole, fraction) = magnitude.split_once('.').ok_or("replay fixed decimal")?;
    if fraction.len() != 5 {
        return Err("replay fixed precision");
    }
    let whole: u64 = whole.parse().map_err(|_| "replay fixed whole integer")?;
    let fraction: u64 = fraction
        .parse()
        .map_err(|_| "replay fixed fraction integer")?;
    if whole > 32768 || fraction >= 100000 {
        return Err("replay fixed range");
    }
    let magnitude = (whole * 65536 + (fraction * 65536).div_ceil(100000)) as i64;
    let raw = if negative { -magnitude } else { magnitude };
    let value = Fixed {
        raw: raw.try_into().map_err(|_| "replay fixed raw range")?,
    };
    if format!("{value:?}") != text {
        return Err("replay fixed is not canonical or lossless");
    }
    Ok(value)
}

fn wrapped_number(text: &str, wrapper: &str) -> Result<u32, &'static str> {
    assert!(["AbilitySlot", "ItemSlot", "ItemId"].contains(&wrapper));
    assert!(text.len() <= 2048);
    text.strip_prefix(&format!("{wrapper}("))
        .and_then(|value| value.strip_suffix(')'))
        .ok_or("replay numeric wrapper")?
        .parse()
        .map_err(|_| "replay wrapped integer")
}

fn item_slot(text: &str) -> Result<ItemSlot, &'static str> {
    Ok(ItemSlot(
        wrapped_number(text, "ItemSlot")?
            .try_into()
            .map_err(|_| "replay item slot range")?,
    ))
}

#[test]
fn matched_contract_replay_fixed_inverts_every_fraction_for_both_signs() {
    for fraction in 0..65536 {
        for raw in [fraction, -fraction, 15000 * 65536 + fraction] {
            let value = Fixed { raw };
            assert_eq!(parse_fixed(&format!("{value:?}")), Ok(value));
        }
    }
    for value in [Fixed::MIN, Fixed::MAX] {
        assert_eq!(parse_fixed(&format!("{value:?}")), Ok(value));
    }
    assert_eq!(parse_fixed("32768.00000"), Err("replay fixed raw range"));
    assert_eq!(
        parse_fixed("0.00002"),
        Err("replay fixed is not canonical or lossless")
    );
}

#[test]
fn matched_contract_replay_request_roundtrips_all_order_and_target_variants() {
    let entity = EntityId {
        idx: 7,
        generation: 3,
    };
    let targets = [
        Target::None,
        Target::Unit(entity),
        Target::Pos(Vec2 {
            x: Fixed { raw: -99999 },
            y: Fixed::MAX,
        }),
    ];
    let mut orders = vec![
        Order::Buy { item: ItemId(33) },
        Order::Learn {
            slot: AbilitySlot(2),
        },
        Order::Sell { slot: ItemSlot(1) },
        Order::Swap {
            from: ItemSlot(1),
            to: ItemSlot(3),
        },
    ];
    for target in targets {
        orders.extend([
            Order::Move { target },
            Order::Attack { target },
            Order::Take { target },
            Order::Cast {
                slot: AbilitySlot(2),
                target,
            },
            Order::Use {
                slot: ItemSlot(1),
                target,
            },
            Order::Put {
                slot: ItemSlot(1),
                target,
            },
        ]);
    }
    assert!(orders.len() <= 32);
    for order in orders {
        for unit in [None, Some(entity)] {
            let request = Some(Request {
                seq: 9,
                unit,
                order,
            });
            assert_eq!(parse_request(&format!("{request:?}")), Ok(request));
        }
    }
    assert_eq!(parse_request("None"), Ok(None));
    assert_eq!(parse_request("corrupt"), Err("replay request envelope"));
    assert_eq!(
        parse_order("Unknown { target: None }"),
        Err("unsupported replay order kind")
    );
}
