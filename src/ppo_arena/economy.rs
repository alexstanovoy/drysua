//! Farming, spending and positioning of one seat over a game, from its own
//! tracker and the events it is shown only.

use bota_proto::{EventKind, Fixed, ItemId, Team, UnitKind, Vec2};
use serde_json::{Value, json};

use crate::StateTracker;
use crate::tracker::map_maximum_raw;

/// Items counted by id; any larger id shares the last bucket.
const ITEM_BUCKETS: usize = 64;
/// A hero this close to its own fountain is healing there, not farming.
const FOUNTAIN_RADIUS: i64 = 1_200;
/// Where a hero spent a game tick, in log and report order.
pub(crate) const PLACE_LABELS: [&str; 5] = ["dead", "fountain", "base", "lane", "enemy_base"];

/// One seat's economy at a moment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EconomyStanding {
    pub last_hits: u16,
    pub denies: u16,
    /// Every gold gain after the first observation, whatever its source.
    pub gold_earned: i32,
    /// Unspent gold plus the whole shop cost of every owned item.
    pub net_worth: i32,
}

impl EconomyStanding {
    pub(crate) fn json(&self) -> Value {
        json!({"last_hits": self.last_hits, "denies": self.denies,
               "gold_earned": self.gold_earned, "net_worth": self.net_worth})
    }
}

/// Gold, items and places one seat accumulated over its game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SeatEconomy {
    gold_earned: i32,
    gold: Option<i32>,
    bought: [u16; ITEM_BUCKETS],
    /// Charges (or single uses) of charged items that left the seat's hands.
    used: [u16; ITEM_BUCKETS],
    charges: [u16; ITEM_BUCKETS],
    /// Game-clock ticks per [`PLACE_LABELS`] entry.
    places: [u32; PLACE_LABELS.len()],
    /// Own fountain, then own and enemy front towers, fixed at first sight.
    landmarks: Option<[Vec2; 3]>,
}

impl Default for SeatEconomy {
    fn default() -> Self {
        Self {
            gold_earned: 0,
            gold: None,
            bought: [0; ITEM_BUCKETS],
            used: [0; ITEM_BUCKETS],
            charges: [0; ITEM_BUCKETS],
            places: [0; PLACE_LABELS.len()],
            landmarks: None,
        }
    }
}

impl SeatEconomy {
    /// Folds one completed tick; the tick's snapshot is already observed.
    pub(super) fn observe(&mut self, tracker: &StateTracker, events: &[EventKind]) {
        for event in events {
            if let EventKind::ItemBought { slot, item } = event
                && *slot == tracker.slot()
            {
                let bucket = &mut self.bought[bucket(*item)];
                *bucket = bucket.saturating_add(1);
            }
        }
        let gold = tracker.own_player().and_then(|player| player.gold);
        if let (Some(previous), Some(gold)) = (self.gold, gold) {
            self.gold_earned = self.gold_earned.saturating_add((gold - previous).max(0));
        }
        self.gold = gold.or(self.gold);
        self.observe_charges(tracker);
        self.observe_place(tracker);
    }

    pub(crate) fn standing(&self, tracker: &StateTracker) -> EconomyStanding {
        let player = tracker.own_player();
        let assets = i32::try_from(crate::feature::own_asset_value(tracker)).unwrap_or(i32::MAX);
        EconomyStanding {
            last_hits: player.map_or(0, |player| player.last_hits),
            denies: player.map_or(0, |player| player.denies),
            gold_earned: self.gold_earned,
            net_worth: self.gold.unwrap_or(0).saturating_add(assets),
        }
    }

    pub(crate) fn items_bought(&self) -> u32 {
        self.bought.iter().map(|&count| u32::from(count)).sum()
    }

    pub(crate) fn consumables_used(&self) -> u32 {
        self.used.iter().map(|&count| u32::from(count)).sum()
    }

    pub(crate) const fn places(&self) -> [u32; PLACE_LABELS.len()] {
        self.places
    }

    /// Items bought and charges used by item id, and ticks per place.
    pub(crate) fn json(&self) -> Value {
        json!({
            "items_bought": by_item(&self.bought),
            "consumables_used": by_item(&self.used),
            "places": PLACE_LABELS
                .iter()
                .zip(self.places)
                .map(|(label, ticks)| ((*label).to_owned(), json!(ticks)))
                .collect::<serde_json::Map<String, Value>>(),
        })
    }

    /// A charged item's charges going down, or one vanishing, counts as used.
    fn observe_charges(&mut self, tracker: &StateTracker) {
        let mut charges = [0u16; ITEM_BUCKETS];
        for item in crate::feature::own_items(tracker) {
            if let Some(count) = item.charges {
                let held = &mut charges[bucket(item.id)];
                *held = held.saturating_add(u16::from(count));
            }
        }
        for ((used, previous), now) in self.used.iter_mut().zip(self.charges).zip(charges) {
            *used = used.saturating_add(previous.saturating_sub(now));
        }
        self.charges = charges;
    }

    fn observe_place(&mut self, tracker: &StateTracker) {
        let Some(view) = tracker.current() else {
            return;
        };
        if view.tick < crate::MAP2_PREGAME_TICKS {
            return;
        }
        if self.landmarks.is_none() {
            self.landmarks = landmarks(tracker);
        }
        let place = match (tracker.own_hero(), self.landmarks) {
            (None, _) => 0,
            (Some(_), None) => return,
            (Some(hero), Some([fountain, own, enemy])) => place(hero.pos, fountain, own, enemy),
        };
        self.places[place] = self.places[place].saturating_add(1);
    }
}

fn bucket(item: ItemId) -> usize {
    usize::from(item.0).min(ITEM_BUCKETS - 1)
}

fn by_item(counts: &[u16; ITEM_BUCKETS]) -> Value {
    counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(id, count)| (id.to_string(), json!(count)))
        .collect::<serde_json::Map<String, Value>>()
        .into()
}

/// The own fountain and the own and enemy towers nearest the map centre: the
/// middle lane's front towers, the ones the game is fought over.
fn landmarks(tracker: &StateTracker) -> Option<[Vec2; 3]> {
    let fountain = crate::scripted::tactics::own_fountain(tracker)?;
    let half = Fixed {
        raw: i32::try_from(map_maximum_raw(tracker.metadata().terrain_cells) / 2).ok()?,
    };
    let centre = Vec2 { x: half, y: half };
    let team = tracker.team();
    let front = |own: bool| {
        tracker
            .current()
            .into_iter()
            .flat_map(|view| &view.units)
            .filter(|unit| {
                unit.kind == UnitKind::Tower
                    && unit.team != Team::Neutral
                    && (unit.team == team) == own
            })
            .min_by_key(|unit| unit.pos.distance_squared(centre))
            .map(|unit| unit.pos)
    };
    Some([fountain, front(true)?, front(false)?])
}

/// Fountain, then the position along the own-to-enemy tower axis: behind the
/// own tower, between the towers, or past the enemy tower.
fn place(hero: Vec2, fountain: Vec2, own: Vec2, enemy: Vec2) -> usize {
    let whole = |point: Vec2| (i64::from(point.x.to_int()), i64::from(point.y.to_int()));
    let (hero, fountain, own, enemy) = (whole(hero), whole(fountain), whole(own), whole(enemy));
    let (dx, dy) = (hero.0 - fountain.0, hero.1 - fountain.1);
    if dx * dx + dy * dy <= FOUNTAIN_RADIUS * FOUNTAIN_RADIUS {
        return 1;
    }
    let axis = (enemy.0 - own.0, enemy.1 - own.1);
    let along = (hero.0 - own.0) * axis.0 + (hero.1 - own.1) * axis.1;
    if along < 0 {
        2
    } else if along <= axis.0 * axis.0 + axis.1 * axis.1 {
        3
    } else {
        4
    }
}
