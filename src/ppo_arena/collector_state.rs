//! Checkpointed collection state: the update a resumed run collects next, its
//! actor weights version and spawn modifiers, and every slot's in-flight game.
//!
//! The encoding is little-endian, versioned, bounded and has no trailing bytes.

use bota_proto::ModifierSpec;

use super::slot::{ActionLog, GamePlan, MAX_SLOTS, OpenInterval, OpponentKind, SlotSnapshot};
use crate::{MAP2_ACTOR_DECISIONS, MAX_COLLECTION_STATE_BYTES, PpoError};

const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CollectorState {
    /// The update whose collection the snapshot starts.
    pub(crate) update: u64,
    /// Completed updates of the actor weights collecting `update`.
    pub(crate) actor_version: u64,
    pub(crate) spec: ModifierSpec,
    /// Every slot, in global slot order.
    pub(crate) slots: Vec<SlotSnapshot>,
}

impl CollectorState {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4096);
        put_u32(&mut bytes, VERSION);
        put_u64(&mut bytes, self.update);
        put_u64(&mut bytes, self.actor_version);
        put_spec(&mut bytes, self.spec);
        put_u32(&mut bytes, self.slots.len() as u32);
        for slot in &self.slots {
            put_slot(&mut bytes, slot);
        }
        assert!(bytes.len() <= MAX_COLLECTION_STATE_BYTES);
        bytes
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, PpoError> {
        if bytes.len() > MAX_COLLECTION_STATE_BYTES {
            return Err(invalid());
        }
        let mut reader = Reader { bytes, offset: 0 };
        if reader.u32()? != VERSION {
            return Err(invalid());
        }
        let update = reader.u64()?;
        let actor_version = reader.u64()?;
        let spec = reader.spec()?;
        let count = reader.u32()? as usize;
        if !(1..=MAX_SLOTS).contains(&count) {
            return Err(invalid());
        }
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            slots.push(reader.slot()?);
        }
        if reader.offset != bytes.len() || actor_version > update {
            return Err(invalid());
        }
        Ok(Self {
            update,
            actor_version,
            spec,
            slots,
        })
    }
}

fn invalid() -> PpoError {
    PpoError::InvalidConfig("collector checkpoint state")
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn spec_fields(spec: ModifierSpec) -> [i32; 11] {
    [
        spec.magic_resist,
        spec.status_resist,
        spec.physical_damage,
        spec.magic_damage,
        spec.pure_damage,
        spec.cooldown_rate,
        spec.mana_cost_rate,
        spec.move_speed,
        spec.max_hp,
        spec.max_mana,
        spec.gold_income,
    ]
}

fn put_spec(bytes: &mut Vec<u8>, spec: ModifierSpec) {
    for field in spec_fields(spec) {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
}

fn put_slot(bytes: &mut Vec<u8>, slot: &SlotSnapshot) {
    let plan = slot.plan;
    put_u32(bytes, plan.slot as u32);
    put_u64(bytes, plan.game);
    put_u64(bytes, plan.start_update);
    put_spec(bytes, plan.spec);
    put_u32(bytes, plan.seat as u32);
    let (kind, index) = match plan.opponent {
        OpponentKind::Teacher => (0, 0),
        OpponentKind::Snapshot(index) => (1, index as u32),
        OpponentKind::SelfPlay => (2, 0),
    };
    put_u32(bytes, kind);
    put_u32(bytes, index);
    put_u32(bytes, plan.decision_cap as u32);
    for (state, draws) in [slot.actor, slot.opponent] {
        put_u64(bytes, state);
        put_u64(bytes, draws);
    }
    match slot.open {
        None => put_u32(bytes, 0),
        Some(open) => {
            put_u32(bytes, 1);
            put_u64(bytes, open.behaviour);
            put_u32(bytes, open.log_probability.to_bits());
            put_u32(bytes, open.value.to_bits());
        }
    }
    for log in [&slot.log.policy, &slot.log.opponent] {
        put_u32(bytes, log.len() as u32);
        for &word in log {
            put_u32(bytes, word);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], PpoError> {
        let end = self.offset.checked_add(N).ok_or_else(invalid)?;
        let slice = self.bytes.get(self.offset..end).ok_or_else(invalid)?;
        self.offset = end;
        Ok(slice.try_into().expect("exact length"))
    }

    fn u32(&mut self) -> Result<u32, PpoError> {
        Ok(u32::from_le_bytes(self.take()?))
    }

    fn u64(&mut self) -> Result<u64, PpoError> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn spec(&mut self) -> Result<ModifierSpec, PpoError> {
        let mut fields = [0i32; 11];
        for field in &mut fields {
            *field = i32::from_le_bytes(self.take()?);
        }
        let [
            magic_resist,
            status_resist,
            physical_damage,
            magic_damage,
            pure_damage,
            cooldown_rate,
            mana_cost_rate,
            move_speed,
            max_hp,
            max_mana,
            gold_income,
        ] = fields;
        Ok(ModifierSpec {
            magic_resist,
            status_resist,
            physical_damage,
            magic_damage,
            pure_damage,
            cooldown_rate,
            mana_cost_rate,
            move_speed,
            max_hp,
            max_mana,
            gold_income,
        })
    }

    fn log(&mut self) -> Result<Vec<u32>, PpoError> {
        let length = self.u32()? as usize;
        if length >= MAP2_ACTOR_DECISIONS {
            return Err(invalid());
        }
        (0..length).map(|_| self.u32()).collect()
    }

    fn slot(&mut self) -> Result<SlotSnapshot, PpoError> {
        let slot = self.u32()? as usize;
        let game = self.u64()?;
        let start_update = self.u64()?;
        let spec = self.spec()?;
        let seat = self.u32()? as usize;
        let opponent = match (self.u32()?, self.u32()? as usize) {
            (0, 0) => OpponentKind::Teacher,
            (1, index) if index < 16 => OpponentKind::Snapshot(index),
            (2, 0) => OpponentKind::SelfPlay,
            _ => return Err(invalid()),
        };
        let decision_cap = self.u32()? as usize;
        if slot >= MAX_SLOTS || seat > 1 || !(1..=MAP2_ACTOR_DECISIONS).contains(&decision_cap) {
            return Err(invalid());
        }
        let actor = (self.u64()?, self.u64()?);
        let opponent_random = (self.u64()?, self.u64()?);
        let open = match self.u32()? {
            0 => None,
            1 => Some(OpenInterval {
                behaviour: self.u64()?,
                log_probability: f32::from_bits(self.u32()?),
                value: f32::from_bits(self.u32()?),
            }),
            _ => return Err(invalid()),
        };
        let log = ActionLog {
            policy: self.log()?,
            opponent: self.log()?,
        };
        Ok(SlotSnapshot {
            plan: GamePlan {
                slot,
                game,
                start_update,
                spec,
                seat,
                opponent,
                decision_cap,
            },
            log,
            actor,
            opponent: opponent_random,
            open,
        })
    }
}
