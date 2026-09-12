use bota_proto::{Team, UnitKind, Vec2};

use super::observation::SnapshotFacts;
use super::{LANE_SCALE, MAX_TOWERS, Map2Reward, Map2RewardError, TOWER_SCALE, limit};

#[derive(Clone, Copy, Debug)]
pub(super) struct Tower {
    pub team: Team,
    pub hp: i32,
    pub maximum: i32,
}

impl Map2Reward {
    pub(super) fn check_tower_capacity(
        &self,
        pending: &SnapshotFacts,
    ) -> Result<(), Map2RewardError> {
        let new = pending
            .units
            .iter()
            .filter(|unit| unit.kind == UnitKind::Tower && !self.towers.contains_key(&unit.id))
            .count();
        limit("tower history", self.towers.len() + new, MAX_TOWERS)
    }

    pub(super) fn update_towers(&mut self, pending: &SnapshotFacts) {
        for unit in pending
            .units
            .iter()
            .filter(|unit| unit.kind == UnitKind::Tower)
        {
            self.towers.insert(
                unit.id,
                Tower {
                    team: unit.team,
                    hp: unit.hp,
                    maximum: unit.maximum,
                },
            );
        }
        assert!(self.towers.len() <= MAX_TOWERS);
        assert!(self.towers.values().all(|tower| tower.maximum > 0));
    }

    pub(super) fn observe_potentials(&mut self, pending: &SnapshotFacts) {
        let tower = self.tower_value();
        let lane = lane_value(pending, self.roles[0].team);
        self.lane_observed = lane.is_some();
        let lane = lane.unwrap_or(self.lane_potential);
        if self.current.is_some() {
            self.interval.tower_health += tower - self.tower_potential;
            self.interval.lane_pressure += lane - self.lane_potential;
            self.interval.observations.lane_observed_ticks += u64::from(self.lane_observed);
        }
        self.tower_potential = tower;
        self.lane_potential = lane;
        assert!(tower.abs() <= TOWER_SCALE + 1.0e-12);
        assert!(lane.abs() <= LANE_SCALE + 1.0e-12);
    }

    fn tower_value(&self) -> f64 {
        let mut total = [0.0; 2];
        let mut count = [0u32; 2];
        for tower in self.towers.values() {
            let Some(index) = self.roles.iter().position(|role| role.team == tower.team) else {
                continue;
            };
            total[index] += f64::from(tower.hp) / f64::from(tower.maximum);
            count[index] += 1;
        }
        if count.contains(&0) {
            return self.tower_potential;
        }
        TOWER_SCALE * (total[0] / f64::from(count[0]) - total[1] / f64::from(count[1]))
    }
}

fn lane_value(pending: &SnapshotFacts, own_team: Team) -> Option<f64> {
    let own = pending
        .units
        .iter()
        .find(|unit| unit.kind == UnitKind::Fountain && unit.team == own_team)?;
    let enemy = pending.units.iter().find(|unit| {
        unit.kind == UnitKind::Fountain && unit.team != own_team && unit.team != Team::Neutral
    })?;
    let axis = [
        f64::from(enemy.pos.x.raw) - f64::from(own.pos.x.raw),
        f64::from(enemy.pos.y.raw) - f64::from(own.pos.y.raw),
    ];
    let squared = axis[0] * axis[0] + axis[1] * axis[1];
    if squared == 0.0 {
        return None;
    }
    let mut sum = [0.0; 2];
    let mut count = [0u32; 2];
    for unit in &pending.units {
        if unit.hp == 0 || !super::events::lane_creep(unit.kind) || unit.team == Team::Neutral {
            continue;
        }
        let index = usize::from(unit.team != own_team);
        sum[index] += axis_progress(unit.pos, own.pos, axis, squared);
        count[index] += 1;
    }
    if count.contains(&0) {
        return None;
    }
    Some(LANE_SCALE * (sum[0] / f64::from(count[0]) + sum[1] / f64::from(count[1]) - 1.0))
}

fn axis_progress(position: Vec2, origin: Vec2, axis: [f64; 2], squared: f64) -> f64 {
    assert!(squared > 0.0);
    assert!(squared.is_finite());
    let offset = [
        f64::from(position.x.raw) - f64::from(origin.x.raw),
        f64::from(position.y.raw) - f64::from(origin.y.raw),
    ];
    ((offset[0] * axis[0] + offset[1] * axis[1]) / squared).clamp(0.0, 1.0)
}
