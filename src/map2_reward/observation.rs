use bota_proto::{EntityId, HeroId, MapId, MatchInfo, SlotId, Team, UnitKind, UnitView, WorldView};

use super::{
    MAP2_REWARD_MAX_UNITS, MAX_AMOUNT, MAX_TOWERS, MAX_XP, Map2RewardError, invalid, limit,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct Role {
    pub slot: SlotId,
    pub team: Team,
    hero: HeroId,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TowerFact {
    pub id: EntityId,
    pub team: Team,
    pub health: f64,
}

/// Public facts of one seat snapshot, indexed by role (0 own, 1 enemy).
#[derive(Clone, Debug)]
pub(super) struct SnapshotFacts {
    pub tick: u32,
    pub xp: [u32; 2],
    pub deaths: [u16; 2],
    /// Scoreboard hero bodies; absent while dead.
    pub heroes: [Option<EntityId>; 2],
    /// HP fraction of each hero body visible in this snapshot.
    pub hero_health: [Option<f64>; 2],
    pub towers: Vec<TowerFact>,
}

pub(super) fn roles(slot: SlotId, info: &MatchInfo) -> Result<[Role; 2], Map2RewardError> {
    if info.map != MapId(2) {
        return invalid("expected map 2");
    }
    if info.picks.len() != 2 {
        return invalid("expected exactly two picks");
    }
    if u32::from(info.tick_rate) != crate::MAP2_TICK_RATE {
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
    if own.team == Team::Neutral || enemy.team == Team::Neutral || own.team == enemy.team {
        return invalid("picks must be on opposing playable teams");
    }
    Ok([own, enemy].map(|pick| Role {
        slot: pick.slot,
        team: pick.team,
        hero: pick.hero,
    }))
}

/// Snapshot facts into a reusable tower buffer, so the steady state allocates nothing.
pub(super) fn snapshot_into(
    view: &WorldView,
    roles: [Role; 2],
    mut towers: Vec<TowerFact>,
) -> Result<SnapshotFacts, Map2RewardError> {
    if view.viewer != Some(roles[0].team) {
        return invalid("Snapshot viewer does not match own team");
    }
    if view.tick == 0 || view.tick > crate::MAP2_TICK_CAP {
        return invalid("Snapshot tick outside 1..=27900");
    }
    limit("visible units", view.units.len(), MAP2_REWARD_MAX_UNITS)?;
    let (xp, deaths, heroes) = scoreboard(view, roles)?;
    let mut hero_health = [None; 2];
    towers.clear();
    for unit in &view.units {
        if !(1..=MAX_AMOUNT).contains(&unit.max_hp) || !(0..=unit.max_hp).contains(&unit.hp) {
            return invalid("unit health outside 0..=maximum<=1000000");
        }
        if let Some(role) = heroes.iter().position(|id| *id == Some(unit.id)) {
            validate_hero(unit, roles[role])?;
            hero_health[role] = Some(fraction(unit));
        } else if unit.kind == UnitKind::Hero && roles.iter().any(|role| role.team == unit.team) {
            return invalid("visible hero is not a public scoreboard body");
        }
        if unit.kind == UnitKind::Tower && roles.iter().any(|role| role.team == unit.team) {
            limit("visible towers", towers.len() + 1, MAX_TOWERS)?;
            towers.push(TowerFact {
                id: unit.id,
                team: unit.team,
                health: fraction(unit),
            });
        }
    }
    if heroes[0].is_some() && hero_health[0].is_none() {
        return invalid("own living hero is missing from Snapshot");
    }
    Ok(SnapshotFacts {
        tick: view.tick,
        xp,
        deaths,
        heroes,
        hero_health,
        towers,
    })
}

type Scoreboard = ([u32; 2], [u16; 2], [Option<EntityId>; 2]);

fn scoreboard(view: &WorldView, roles: [Role; 2]) -> Result<Scoreboard, Map2RewardError> {
    if view.players.len() != 2 {
        return invalid("expected exactly two scoreboard players");
    }
    let mut xp = [0; 2];
    let mut deaths = [0; 2];
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
        xp[index] = u32::try_from(player.xp).expect("validated scoreboard XP");
        deaths[index] = player.deaths;
        heroes[index] = player.unit;
    }
    if heroes[0].is_some() && heroes[0] == heroes[1] {
        return invalid("opposing heroes share one handle");
    }
    Ok((xp, deaths, heroes))
}

fn validate_hero(unit: &UnitView, role: Role) -> Result<(), Map2RewardError> {
    if unit.kind != UnitKind::Hero
        || unit.team != role.team
        || unit.owner != Some(role.slot)
        || unit.hero != Some(role.hero)
    {
        return invalid("visible hero role differs from public scoreboard");
    }
    Ok(())
}

fn fraction(unit: &UnitView) -> f64 {
    let health = f64::from(unit.hp) / f64::from(unit.max_hp);
    assert!((0.0..=1.0).contains(&health));
    health
}
