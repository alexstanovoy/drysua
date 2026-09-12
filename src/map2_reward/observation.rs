use bota_proto::{
    EntityId, HeroId, MapId, MatchInfo, SlotId, Team, UnitKind, UnitView, Vec2, WorldView,
};

use super::{
    IDENTITY_AGE, MAP2_REWARD_MAX_IDENTITIES, MAP2_REWARD_MAX_UNITS, MAX_AMOUNT, MAX_TICK, MAX_XP,
    Map2Reward, Map2RewardError, invalid, limit,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct Role {
    pub slot: SlotId,
    pub team: Team,
    hero: HeroId,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Identity {
    pub kind: UnitKind,
    pub team: Team,
    pub role: Option<usize>,
    pub seen: u32,
    pub dead: bool,
    pub death_recorded: bool,
}

impl Identity {
    fn retained(self, tick: u32) -> bool {
        self.role.is_some() || tick.saturating_sub(self.seen) <= IDENTITY_AGE
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct UnitFact {
    pub id: EntityId,
    pub kind: UnitKind,
    pub team: Team,
    pub hp: i32,
    pub maximum: i32,
    pub pos: Vec2,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Mana {
    pub id: EntityId,
    pub mana: i32,
    pub maximum: i32,
}

#[derive(Clone, Debug)]
pub(super) struct SnapshotFacts {
    pub tick: u32,
    pub xp: [u32; 2],
    pub heroes: [Option<EntityId>; 2],
    pub mana: Option<Mana>,
    pub units: Vec<UnitFact>,
}

pub(super) fn roles(slot: SlotId, info: &MatchInfo) -> Result<[Role; 2], Map2RewardError> {
    if info.map != MapId(2) {
        return invalid("expected map 2");
    }
    if info.picks.len() != 2 {
        return invalid("expected exactly two picks");
    }
    if info.tick_rate != 30 {
        return invalid("expected 30 simulation ticks per second");
    }
    let own = info
        .picks
        .iter()
        .find(|pick| pick.slot == slot)
        .ok_or(Map2RewardError::Invalid("own slot is absent from picks"))?;
    let enemy = info
        .picks
        .iter()
        .find(|pick| pick.slot != slot)
        .ok_or(Map2RewardError::Invalid("picks contain duplicate slots"))?;
    if own.slot.0 >= 10 || enemy.slot.0 >= 10 {
        return invalid("pick slot is outside 0..10");
    }
    if own.team == Team::Neutral || enemy.team == Team::Neutral || own.team == enemy.team {
        return invalid("picks must be on opposing playable teams");
    }
    Ok([
        Role {
            slot: own.slot,
            team: own.team,
            hero: own.hero,
        },
        Role {
            slot: enemy.slot,
            team: enemy.team,
            hero: enemy.hero,
        },
    ])
}

pub(super) fn snapshot(
    view: &WorldView,
    roles: [Role; 2],
) -> Result<SnapshotFacts, Map2RewardError> {
    if view.viewer != Some(roles[0].team) {
        return invalid("Snapshot viewer does not match own team");
    }
    if view.tick == 0 || view.tick > MAX_TICK {
        return invalid("Snapshot tick outside 1..=3600000");
    }
    limit("visible units", view.units.len(), MAP2_REWARD_MAX_UNITS)?;
    limit("projectiles", view.projectiles.len(), 4096)?;
    limit("felled trees", view.felled_trees.len(), 4096)?;
    limit("planted trees", view.planted_trees.len(), 4096)?;
    limit("loot", view.loot.len(), 16)?;
    let (xp, heroes) = scoreboard(view, roles)?;
    if view.units.windows(2).any(|pair| pair[0].id >= pair[1].id) {
        return invalid("visible units must have strictly increasing handles");
    }
    let mut mana = None;
    let mut units = Vec::with_capacity(view.units.len());
    for unit in &view.units {
        validate_unit(unit)?;
        if unit.kind == UnitKind::Hero {
            validate_hero(unit, heroes, roles)?;
        } else if heroes.contains(&Some(unit.id)) {
            return invalid("scoreboard body is not a hero");
        }
        if Some(unit.id) == heroes[0] {
            mana = Some(Mana {
                id: unit.id,
                mana: unit.mana,
                maximum: unit.max_mana,
            });
        }
        units.push(UnitFact {
            id: unit.id,
            kind: unit.kind,
            team: unit.team,
            hp: unit.hp,
            maximum: unit.max_hp,
            pos: unit.pos,
        });
    }
    if heroes[0].is_some() && mana.is_none() {
        return invalid("own living hero is missing from Snapshot");
    }
    assert!(units.len() <= MAP2_REWARD_MAX_UNITS);
    assert!(xp.iter().all(|value| *value <= MAX_XP as u32));
    Ok(SnapshotFacts {
        tick: view.tick,
        xp,
        heroes,
        mana,
        units,
    })
}

fn scoreboard(
    view: &WorldView,
    roles: [Role; 2],
) -> Result<([u32; 2], [Option<EntityId>; 2]), Map2RewardError> {
    if view.players.len() != 2 {
        return invalid("expected exactly two scoreboard players");
    }
    if view.players[0].slot >= view.players[1].slot {
        return invalid("scoreboard slots must be strictly increasing");
    }
    let mut xp = [0; 2];
    let mut heroes = [None; 2];
    for (index, role) in roles.iter().enumerate() {
        let player = view
            .players
            .iter()
            .find(|player| player.slot == role.slot)
            .ok_or(Map2RewardError::Invalid(
                "scoreboard is missing a picked slot",
            ))?;
        if player.team != role.team || player.hero != role.hero {
            return invalid("scoreboard role differs from public pick");
        }
        if !(0..=MAX_XP).contains(&player.xp) {
            return invalid("scoreboard XP outside 0..=1000000000");
        }
        if index == 0 && player.gold.is_none_or(|gold| gold < 0) {
            return invalid("own gold is absent or negative");
        }
        if index == 1 && (player.gold.is_some() || player.stash.is_some() || player.kit.is_some()) {
            return invalid("opposing private scoreboard fields are present");
        }
        if let Some(stash) = &player.stash {
            limit("stash slots", stash.len(), 6)?;
        }
        if let Some(kit) = &player.kit {
            limit("dead hero ability slots", kit.abilities.len(), 8)?;
            limit("dead hero item slots", kit.items.len(), 9)?;
        }
        xp[index] = player.xp as u32;
        heroes[index] = player.unit;
    }
    if heroes[0].is_some() && heroes[0] == heroes[1] {
        return invalid("opposing heroes share one handle");
    }
    Ok((xp, heroes))
}

fn validate_unit(unit: &UnitView) -> Result<(), Map2RewardError> {
    if !(1..=MAX_AMOUNT).contains(&unit.max_hp) || !(0..=unit.max_hp).contains(&unit.hp) {
        return invalid("unit health outside 0..=maximum<=1000000");
    }
    if !(0..=MAX_AMOUNT).contains(&unit.max_mana) || !(0..=unit.max_mana).contains(&unit.mana) {
        return invalid("unit mana outside 0..=maximum<=1000000");
    }
    limit("ability slots", unit.abilities.len(), 8)?;
    limit("item slots", unit.items.len(), 9)?;
    limit("unit effects", unit.effects.len(), 32)?;
    Ok(())
}

fn validate_hero(
    unit: &UnitView,
    heroes: [Option<EntityId>; 2],
    roles: [Role; 2],
) -> Result<(), Map2RewardError> {
    let Some(index) = heroes.iter().position(|id| *id == Some(unit.id)) else {
        return invalid("visible hero is not a public scoreboard body");
    };
    if unit.team != roles[index].team
        || unit.owner != Some(roles[index].slot)
        || unit.hero != Some(roles[index].hero)
    {
        return invalid("visible hero role differs from public scoreboard");
    }
    Ok(())
}

impl Map2Reward {
    pub(super) fn check_snapshot_progress(
        &self,
        pending: &SnapshotFacts,
    ) -> Result<(), Map2RewardError> {
        if let Some(current) = &self.current
            && pending
                .xp
                .iter()
                .zip(current.xp)
                .any(|(next, before)| *next < before)
        {
            return invalid("public cumulative XP decreased");
        }
        for unit in &pending.units {
            if let Some(previous) = self.identities.get(&unit.id)
                && previous.dead
                && unit.hp > 0
            {
                return invalid("dead unit reappeared without a new generation");
            }
            if let Some(previous) = self.identities.get(&unit.id)
                && (unit.kind != previous.kind || unit.team != previous.team)
            {
                return invalid("unit kind or team changed without a new generation");
            }
        }
        for (role, id) in pending.heroes.iter().enumerate() {
            if let Some(identity) = id.and_then(|id| self.identities.get(&id))
                && (identity.kind != UnitKind::Hero
                    || identity.team != self.roles[role].team
                    || identity.dead
                    || identity.role.is_some_and(|previous| previous != role))
            {
                return invalid("scoreboard body conflicts with public identity history");
            }
        }
        Ok(())
    }

    pub(super) fn check_identity_capacity(
        &self,
        pending: &SnapshotFacts,
    ) -> Result<(), Map2RewardError> {
        let known = |id: &EntityId| {
            self.identities
                .get(id)
                .is_some_and(|identity| identity.retained(pending.tick))
        };
        let retained = self
            .identities
            .values()
            .filter(|identity| identity.retained(pending.tick))
            .count();
        let new_units = pending.units.iter().filter(|unit| !known(&unit.id)).count();
        let new_heroes = pending
            .heroes
            .iter()
            .flatten()
            .filter(|id| {
                !known(id)
                    && pending
                        .units
                        .binary_search_by_key(*id, |unit| unit.id)
                        .is_err()
            })
            .count();
        limit(
            "identity history",
            retained + new_units + new_heroes,
            MAP2_REWARD_MAX_IDENTITIES,
        )
    }

    pub(super) fn update_identities(&mut self, pending: &SnapshotFacts) {
        self.identities
            .retain(|_, identity| identity.retained(pending.tick));
        for unit in &pending.units {
            let identity = self.identities.entry(unit.id).or_insert(Identity {
                kind: unit.kind,
                team: unit.team,
                role: None,
                seen: pending.tick,
                dead: false,
                death_recorded: false,
            });
            identity.seen = pending.tick;
        }
        for (role, id) in pending.heroes.iter().enumerate() {
            let Some(id) = id else {
                continue;
            };
            let identity = self.identities.entry(*id).or_insert(Identity {
                kind: UnitKind::Hero,
                team: self.roles[role].team,
                role: Some(role),
                seen: pending.tick,
                dead: false,
                death_recorded: false,
            });
            identity.role = Some(role);
            identity.seen = pending.tick;
        }
        assert!(self.identities.len() <= MAP2_REWARD_MAX_IDENTITIES);
        assert!(
            pending
                .heroes
                .iter()
                .flatten()
                .all(|id| self.identities.contains_key(id))
        );
    }
}
