use bota_proto::{ItemId, ItemSlot, Order};

use crate::tracker::HERO_ITEM_SLOTS;
use crate::{ActionSpace, ControlledUnit, IssuedOrder};

/// Ticks an item stays muted after leaving the backpack in the drysua schema.
pub const BACKPACK_MUTE_TICKS: u32 = 180;
/// Town Portal Scroll in the drysua item schema.
pub const TOWN_PORTAL_SCROLL: ItemId = ItemId(8);
/// Body-wide Town Portal Scroll wait in ticks.
pub const TOWN_PORTAL_WAIT_TICKS: u32 = 2_100;
/// Items with a body-wide wait and their wait in ticks.
pub const SHARED_WAITS: [(ItemId, u32); 1] = [(TOWN_PORTAL_SCROLL, TOWN_PORTAL_WAIT_TICKS)];
/// Maximum recent timer replacements retained for exact rejection rollback.
///
/// Rejection is supported for these retained requests. An older effective
/// timer is retained as a base but its already-evicted sequence is not
/// rejectable under the one-outstanding-request protocol.
pub const MAX_READINESS_TIMER_HISTORY: usize = 8;

const INVENTORY_SLOTS: usize = 6;
const BACKPACK_SLOT_START: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReadinessTimer {
    sequence: u32,
    ready_tick: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TimerJournal {
    base: Option<ReadinessTimer>,
    entries: [Option<ReadinessTimer>; MAX_READINESS_TIMER_HISTORY],
    count: usize,
}

impl TimerJournal {
    const fn new() -> Self {
        Self {
            base: None,
            entries: [None; MAX_READINESS_TIMER_HISTORY],
            count: 0,
        }
    }

    fn push(&mut self, timer: ReadinessTimer) {
        if self.count == MAX_READINESS_TIMER_HISTORY {
            self.base = self.entries[0];
            self.entries.copy_within(1..MAX_READINESS_TIMER_HISTORY, 0);
            self.count -= 1;
        }
        self.entries[self.count] = Some(timer);
        self.count += 1;
    }

    fn reject(&mut self, sequence: u32) -> bool {
        let mut kept = [None; MAX_READINESS_TIMER_HISTORY];
        let mut count = 0usize;
        let mut removed = false;
        for timer in self.entries[..self.count].iter().flatten().copied() {
            if timer.sequence == sequence {
                removed = true;
            } else {
                kept[count] = Some(timer);
                count += 1;
            }
        }
        self.entries = kept;
        self.count = count;
        removed
    }

    fn current(&self) -> Option<ReadinessTimer> {
        self.entries[..self.count]
            .iter()
            .rev()
            .find_map(|timer| *timer)
            .or(self.base)
    }
}

/// Locally tracked item timers that the wire does not report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemReadiness {
    hero_mutes: [TimerJournal; INVENTORY_SLOTS],
    shared_waits: [TimerJournal; 2],
}

impl ItemReadiness {
    /// Empty timer state for a fresh match.
    pub const fn new() -> Self {
        Self {
            hero_mutes: [TimerJournal::new(); INVENTORY_SLOTS],
            shared_waits: [TimerJournal::new(); 2],
        }
    }

    /// Records timers implied by one sent order against the space it decoded from.
    pub fn note_sent(&mut self, sequence: u32, issued: IssuedOrder, space: &ActionSpace) {
        let apply_tick = space.tick().saturating_add(1);
        match issued.order {
            Order::Swap { from, to } => {
                self.note_swap(sequence, issued, from, to, apply_tick);
            }
            Order::Use { slot, .. } => {
                self.note_use(sequence, issued, space, slot, apply_tick);
            }
            _ => {}
        }
    }

    /// Removes every timer created by the rejected sequence.
    pub fn note_rejected(&mut self, sequence: u32) -> bool {
        let mut removed = false;
        for journal in self
            .hero_mutes
            .iter_mut()
            .chain(self.shared_waits.iter_mut())
        {
            removed |= journal.reject(sequence);
        }
        removed
    }

    /// Whether an inventory slot is still muted at a snapshot tick.
    pub fn inventory_muted(&self, unit: ControlledUnit, slot: ItemSlot, tick: u32) -> bool {
        self.inventory_mute_left(unit, slot, tick)
            .is_some_and(|left| left > 0)
    }

    /// Remaining local inventory mute ticks, absent where mute is not applicable.
    pub fn inventory_mute_left(
        &self,
        unit: ControlledUnit,
        slot: ItemSlot,
        tick: u32,
    ) -> Option<u32> {
        if unit != ControlledUnit::Hero {
            return None;
        }
        let journal = self.hero_mutes.get(usize::from(slot.0))?;
        Some(
            journal
                .current()
                .map_or(0, |mute| mute.ready_tick.saturating_sub(tick)),
        )
    }

    /// Whether a body still owes a shared wait for an item at a snapshot tick.
    pub fn shared_waiting(&self, unit: ControlledUnit, item: ItemId, tick: u32) -> bool {
        self.shared_wait_left(unit, item, tick)
            .is_some_and(|left| left > 0)
    }

    /// Remaining body-wide item wait ticks, absent for items without such a wait.
    pub fn shared_wait_left(&self, unit: ControlledUnit, item: ItemId, tick: u32) -> Option<u32> {
        if !SHARED_WAITS.iter().any(|(shared, _)| *shared == item) {
            return None;
        }
        let journal = self.shared_waits[unit.index()];
        Some(
            journal
                .current()
                .map_or(0, |wait| wait.ready_tick.saturating_sub(tick)),
        )
    }

    fn note_swap(
        &mut self,
        sequence: u32,
        issued: IssuedOrder,
        from: ItemSlot,
        to: ItemSlot,
        apply_tick: u32,
    ) {
        // A courier bag has no backpack, and stash moves never mute.
        if issued.unit.is_some() {
            return;
        }
        let from = usize::from(from.0);
        let to = usize::from(to.0);
        if !(BACKPACK_SLOT_START..HERO_ITEM_SLOTS).contains(&from) || to >= INVENTORY_SLOTS {
            return;
        }
        self.hero_mutes[to].push(ReadinessTimer {
            sequence,
            ready_tick: apply_tick.saturating_add(BACKPACK_MUTE_TICKS),
        });
    }

    fn note_use(
        &mut self,
        sequence: u32,
        issued: IssuedOrder,
        space: &ActionSpace,
        slot: ItemSlot,
        apply_tick: u32,
    ) {
        let unit = controlled_unit(issued);
        let Some(item) = space.controlled_item(unit, slot) else {
            return;
        };
        if item.id != TOWN_PORTAL_SCROLL {
            return;
        }
        self.shared_waits[unit.index()].push(ReadinessTimer {
            sequence,
            ready_tick: apply_tick.saturating_add(TOWN_PORTAL_WAIT_TICKS),
        });
    }
}

impl Default for ItemReadiness {
    fn default() -> Self {
        Self::new()
    }
}

const fn controlled_unit(issued: IssuedOrder) -> ControlledUnit {
    if issued.unit.is_some() {
        ControlledUnit::Courier
    } else {
        ControlledUnit::Hero
    }
}
