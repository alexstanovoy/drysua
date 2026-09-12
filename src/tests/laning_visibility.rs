use std::collections::BTreeMap;

use super::*;

/// Compares independently observed event multisets; never feeds one seat the other's events.
#[derive(Default)]
pub(super) struct VisibilityAudit {
    heroes: [BTreeMap<EntityId, usize>; 2],
    outgoing: [u64; 2],
    incoming: [u64; 2],
    shared: [u64; 2],
    attacker_only: [u64; 2],
    victim_only: [u64; 2],
    first_difference: Option<String>,
}

impl VisibilityAudit {
    pub(super) fn observe(&mut self, messages: &[Vec<ServerMsg>]) {
        assert_eq!(messages.len(), 2);
        for (side, stream) in messages.iter().enumerate() {
            let view = snapshot(stream);
            for player in &view.players {
                if let Some(hero) = player.unit {
                    self.heroes[side].insert(hero, usize::from(player.slot.0));
                }
            }
            assert!(self.heroes[side].len() <= 128);
        }
        for source in 0..2 {
            let target = 1 - source;
            let attacker = hits(&messages[source], &self.heroes[source], source, target);
            let mut victim = hits(&messages[target], &self.heroes[target], source, target);
            self.outgoing[source] += sum(&attacker);
            self.incoming[target] += sum(&victim);
            for event in attacker {
                if let Some(index) = victim.iter().position(|other| other == &event) {
                    self.shared[source] += amount(&event);
                    victim.swap_remove(index);
                } else {
                    self.attacker_only[source] += amount(&event);
                    self.note_difference(
                        snapshot(&messages[source]).tick,
                        source,
                        "attacker",
                        &event,
                    );
                }
            }
            for event in victim {
                self.victim_only[source] += amount(&event);
                self.note_difference(snapshot(&messages[target]).tick, source, "victim", &event);
            }
        }
    }

    pub(super) fn assert_observers(&self, observers: &[LaningObserver; 2]) {
        for (side, observer) in observers.iter().enumerate() {
            assert_eq!(observer.metrics.hero_damage, self.outgoing[side]);
            assert_eq!(observer.metrics.hero_damage_taken, self.incoming[side]);
            assert_eq!(
                self.outgoing[side],
                self.shared[side] + self.attacker_only[side]
            );
            assert_eq!(
                self.incoming[1 - side],
                self.shared[side] + self.victim_only[side]
            );
            assert_eq!(
                observer.metrics.hero_damage + self.victim_only[side],
                observers[1 - side].metrics.hero_damage_taken + self.attacker_only[side]
            );
        }
    }

    pub(super) fn report(&self, seed: u64, seat: usize, proxy: LaningProxy) {
        if let Some(first) = &self.first_difference {
            println!(
                "laning_visibility seed={seed} seat={seat} proxy={proxy:?} outgoing={:?} incoming={:?} shared={:?} attacker_only={:?} victim_only={:?} first={first}",
                self.outgoing, self.incoming, self.shared, self.attacker_only, self.victim_only
            );
        }
    }

    fn note_difference(&mut self, tick: u32, source: usize, receiver: &str, event: &EventKind) {
        if self.first_difference.is_none() {
            self.first_difference = Some(format!(
                "tick={tick} source_seat={source} receiver={receiver} event={event:?}"
            ));
        }
    }
}

fn snapshot(messages: &[ServerMsg]) -> &WorldView {
    assert!(messages.len() <= 4);
    messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view),
            _ => None,
        })
        .expect("full native seat tick")
}

fn hits(
    messages: &[ServerMsg],
    heroes: &BTreeMap<EntityId, usize>,
    source: usize,
    target: usize,
) -> Vec<EventKind> {
    assert_ne!(source, target);
    let mut output = Vec::new();
    for message in messages {
        if let ServerMsg::Events { events, .. } = message {
            assert!(events.len() <= 4096);
            for event in events {
                if let EventKind::Damaged {
                    source: Some(from),
                    target: into,
                    amount,
                    ..
                } = event
                    && *amount > 0
                    && heroes.get(from) == Some(&source)
                    && heroes.get(into) == Some(&target)
                {
                    output.push(event.clone());
                }
            }
        }
    }
    assert!(output.len() <= 4096);
    output
}

fn amount(event: &EventKind) -> u64 {
    let EventKind::Damaged { amount, .. } = event else {
        panic!("filtered damage")
    };
    u64::try_from(*amount).expect("positive damage")
}

fn sum(events: &[EventKind]) -> u64 {
    assert!(events.len() <= 4096);
    events.iter().map(amount).sum()
}

#[test]
fn native_damage_visibility_explains_39hp_cross_seat_difference_without_unioning_events() {
    for visible in [false, true] {
        let (mut arena, _) = Arena::new(ArenaConfig {
            seats: 2,
            map: MapId(1),
            seed: 9_240_101,
        })
        .expect("historical geometry");
        let baseline = arena.configure_for_test(|world| {
            let source = world.seats[0].unit.expect("source");
            let target = world.seats[1].unit.expect("target");
            world.transform.get_mut(source).expect("position").pos = Vec2::from_ints(
                if visible { 11_900 } else { 2_500 },
                if visible { 9_000 } else { 2_500 },
            );
            world.transform.get_mut(target).expect("position").pos = Vec2::from_ints(12_000, 9_000);
            world.push_hit(Some(source), target, 39, DamageKind::Pure);
        });
        let mut observers = [
            LaningObserver::new(SlotId(0)),
            LaningObserver::new(SlotId(1)),
        ];
        let mut audit = VisibilityAudit::default();
        observe_pair(&mut audit, &mut observers, &baseline.messages);
        let messages = arena
            .step(&[None, None])
            .expect("single damage tick")
            .messages;
        observe_pair(&mut audit, &mut observers, &messages);
        assert_eq!(
            observers[0].metrics.hero_damage,
            if visible { 39 } else { 0 }
        );
        assert_eq!(observers[1].metrics.hero_damage_taken, 39);
        assert_eq!(audit.victim_only[0], if visible { 0 } else { 39 });
        assert_eq!(audit.shared[0], if visible { 39 } else { 0 });
        audit.report(9_240_101, 0, LaningProxy::RightClick);
    }
}

fn observe_pair(
    audit: &mut VisibilityAudit,
    observers: &mut [LaningObserver; 2],
    messages: &[Vec<ServerMsg>],
) {
    audit.observe(messages);
    for (observer, stream) in observers.iter_mut().zip(messages) {
        for message in stream {
            observer.observe(message).expect("seat's own stream only");
        }
    }
    audit.assert_observers(observers);
}
