use bota_proto::{EntityId, EventKind, UnitKind};

use super::{MAP2_REWARD_MAX_EVENTS, MAX_AMOUNT, Map2Reward, Map2RewardError, invalid, limit};

pub(super) fn validate(events: &[EventKind]) -> Result<(), Map2RewardError> {
    limit("event batch", events.len(), MAP2_REWARD_MAX_EVENTS)?;
    for event in events {
        match event {
            EventKind::Damaged { amount, .. } if !(0..=MAX_AMOUNT).contains(amount) => {
                return invalid("damage amount outside 0..=1000000");
            }
            EventKind::Healed { amount, .. } if !(0..=MAX_AMOUNT).contains(amount) => {
                return invalid("healing amount outside 0..=1000000");
            }
            EventKind::Died { gold, denied, .. }
                if !(0..=MAX_AMOUNT).contains(gold) || (*denied && *gold != 0) =>
            {
                return invalid("death gold outside 0..=1000000 or nonzero on deny");
            }
            _ => {}
        }
    }
    Ok(())
}

impl Map2Reward {
    pub(super) fn validate_event_lifecycle(
        &self,
        pending: &super::SnapshotFacts,
        events: &[EventKind],
    ) -> Result<(), Map2RewardError> {
        for event in events {
            let (id, message) = match *event {
                EventKind::Died { unit, .. } => {
                    if pending.heroes.contains(&Some(unit)) {
                        return invalid("dead hero remains in public scoreboard");
                    }
                    (unit, "Died victim is still alive in matching Snapshot")
                }
                EventKind::StructureDestroyed { unit, team } => {
                    self.validate_structure_identity(pending, unit, team)?;
                    (
                        unit,
                        "destroyed structure is still alive in matching Snapshot",
                    )
                }
                _ => continue,
            };
            if let Ok(index) = pending.units.binary_search_by_key(&id, |unit| unit.id)
                && pending.units[index].hp > 0
            {
                return invalid(message);
            }
        }
        Ok(())
    }

    fn validate_structure_identity(
        &self,
        pending: &super::SnapshotFacts,
        unit: EntityId,
        team: bota_proto::Team,
    ) -> Result<(), Map2RewardError> {
        if pending.heroes.contains(&Some(unit)) {
            return invalid("StructureDestroyed names a public hero body");
        }
        let current = pending
            .units
            .binary_search_by_key(&unit, |unit| unit.id)
            .ok()
            .map(|index| (pending.units[index].kind, pending.units[index].team));
        let previous = self
            .identities
            .get(&unit)
            .map(|identity| (identity.kind, identity.team));
        let tower = self
            .towers
            .get(&unit)
            .map(|tower| (UnitKind::Tower, tower.team));
        for (kind, side) in [current, previous, tower].into_iter().flatten() {
            if side != team
                || !matches!(
                    kind,
                    UnitKind::Tower | UnitKind::Ancient | UnitKind::Barracks | UnitKind::Fountain
                )
            {
                return invalid("StructureDestroyed conflicts with public unit identity");
            }
        }
        Ok(())
    }

    pub(super) fn observe_event(&mut self, event: &EventKind) {
        match *event {
            EventKind::Damaged {
                source,
                target,
                amount,
                ..
            } if amount > 0 => {
                self.observe_damage(source, target, amount as u64);
            }
            EventKind::Died {
                unit,
                killer,
                denied,
                gold,
            } => {
                self.observe_death(unit, killer, denied, gold as u64);
            }
            EventKind::StructureDestroyed { unit, team } => {
                if let Some(identity) = self.identities.get_mut(&unit) {
                    identity.dead = true;
                }
                if let Some(tower) = self.towers.get_mut(&unit)
                    && tower.team == team
                {
                    tower.hp = 0;
                }
            }
            _ => {}
        }
    }

    fn observe_damage(&mut self, source: Option<EntityId>, target: EntityId, amount: u64) {
        assert!(amount > 0);
        assert!(amount <= MAX_AMOUNT as u64);
        let environmental = source.is_none();
        let source = source.and_then(|id| self.identities.get(&id)).copied();
        let target = self.identities.get(&target).copied();
        if target.is_none() || (!environmental && source.is_none()) {
            self.interval.observations.unattributed_damage_events += 1;
        }
        let Some(target) = target else {
            return;
        };
        if target.role == Some(0) {
            self.observe_received(source, amount, environmental);
        }
        if source.is_some_and(|source| source.role == Some(0)) && target.role == Some(1) {
            self.interval.observations.hero_damage_dealt += amount;
            self.interval.hero_damage += self.charge(4, amount);
        }
    }

    fn observe_received(
        &mut self,
        source: Option<super::Identity>,
        amount: u64,
        environmental: bool,
    ) {
        let channel = match source {
            Some(source) if source.role == Some(1) => {
                self.interval.observations.hero_damage_taken += amount;
                5
            }
            Some(source) if creep(source.kind) => {
                self.interval.observations.creep_damage_taken += amount;
                6
            }
            Some(_) => {
                self.interval.observations.other_damage_taken += amount;
                7
            }
            None if environmental => {
                self.interval.observations.other_damage_taken += amount;
                7
            }
            None => {
                self.interval.observations.unattributed_damage_taken += amount;
                7
            }
        };
        let cost = self.charge(channel, amount);
        match channel {
            5 => self.interval.hero_damage_taken -= cost,
            6 => self.interval.creep_damage_taken -= cost,
            7 => self.interval.other_damage_taken -= cost,
            _ => unreachable!("received damage has three validated source classes"),
        }
    }

    fn observe_death(&mut self, unit: EntityId, killer: Option<EntityId>, denied: bool, gold: u64) {
        let killer = killer.and_then(|id| self.identities.get(&id)).copied();
        let Some(victim) = self.identities.get_mut(&unit) else {
            self.interval.observations.unattributed_deaths += 1;
            return;
        };
        if victim.death_recorded {
            self.interval.observations.duplicate_deaths += 1;
            return;
        }
        victim.dead = true;
        victim.death_recorded = true;
        let victim = *victim;
        if let Some(tower) = self.towers.get_mut(&unit) {
            tower.hp = 0;
        }
        if denied || gold == 0 {
            return;
        }
        let Some(killer) = killer.filter(|killer| killer.role.is_some()) else {
            self.interval.observations.unattributed_deaths += 1;
            return;
        };
        if killer.team == victim.team {
            return;
        }
        if killer.role == Some(0) {
            self.interval.observations.own_gold_earned += gold;
            self.interval.gold += self.charge(0, gold);
            if victim.kind == UnitKind::CreepNeutral {
                self.interval.observations.neutral_last_hits += 1;
            } else if lane_creep(victim.kind) {
                self.interval.observations.lane_last_hits += 1;
            }
        } else {
            self.interval.observations.enemy_gold_earned += gold;
            self.interval.gold -= self.charge(1, gold);
        }
    }
}

pub(super) fn lane_creep(kind: UnitKind) -> bool {
    matches!(
        kind,
        UnitKind::CreepMelee
            | UnitKind::CreepFlagbearer
            | UnitKind::CreepRanged
            | UnitKind::CreepSiege
    )
}

fn creep(kind: UnitKind) -> bool {
    lane_creep(kind) || kind == UnitKind::CreepNeutral
}
