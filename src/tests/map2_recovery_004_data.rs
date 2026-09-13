use super::*;

pub(super) struct Dataset {
    pub ordinary: [Vec<Row>; 2],
    pub journeys: [Vec<Row>; 8],
    pub counterfactual: Vec<Row>,
}

pub(super) fn collect(report: &mut String) -> Dataset {
    let reference = parent();
    let start = initial();
    let identity = reference.policy_identity().unwrap();
    let ordinary = ordinary_rows(report);
    let journeys = journey_rows(report);
    let counterfactual = counterfactual_rows(&reference, &start, report);
    let dataset = Dataset {
        ordinary,
        journeys,
        counterfactual,
    };
    assert!(dataset.count() <= 4096);
    assert!(dataset.groups().iter().all(|group| !group.is_empty()));
    assert_eq!(identity, reference.policy_identity().unwrap());
    dataset
}

impl Dataset {
    pub fn groups(&self) -> Vec<&Vec<Row>> {
        self.ordinary
            .iter()
            .chain(self.journeys.iter())
            .chain(std::iter::once(&self.counterfactual))
            .collect()
    }
    pub fn count(&self) -> usize {
        self.groups().iter().map(|rows| rows.len()).sum()
    }
    pub fn hashes(&self) -> Vec<String> {
        self.groups().iter().map(|rows| data_hash(rows)).collect()
    }
    pub fn batch<'a>(&'a self, step: u64, random: &mut PpoRng) -> Vec<&'a CheckedActionSet> {
        assert!((1..=512).contains(&step));
        let mut batch = Vec::with_capacity(16);
        for group in &self.ordinary {
            for _ in 0..4 {
                batch.push(draw(group, random));
            }
        }
        let side = (step as usize - 1) % 2;
        for phase in 0..4 {
            batch.push(draw(&self.journeys[side * 4 + phase], random));
        }
        for _ in 0..4 {
            batch.push(draw(&self.counterfactual, random));
        }
        assert_eq!(batch.len(), 16);
        batch
    }
}

fn draw<'a>(rows: &'a [Row], random: &mut PpoRng) -> &'a CheckedActionSet {
    assert!(!rows.is_empty());
    assert!(rows.len() <= 4096);
    &rows[random.below(rows.len() as u64).unwrap() as usize].set
}

fn ordinary_rows(report: &mut String) -> [Vec<Row>; 2] {
    let mut rows = Vec::new();
    for seed in [10102980, 10102981] {
        for side in 0..2 {
            rehearsal_game(seed, side, &mut rows, report);
        }
    }
    let mut groups: [Vec<Row>; 2] = Default::default();
    for row in rows {
        let tick: u32 = row.identity.rsplit_once("tick").unwrap().1.parse().unwrap();
        groups[usize::from(tick > 901)].push(row);
    }
    assert!(groups.iter().all(|group| !group.is_empty()));
    writeln!(
        report,
        "ordinary_early_late={:?}",
        groups.each_ref().map(Vec::len)
    )
    .unwrap();
    groups
}

fn phase(state: &Game, space: &ActionSpace, action: StructuredAction) -> usize {
    let hero = state.seats[state.side].tracker.own_hero().unwrap();
    let close_finish = space.entity_candidates().iter().any(|unit| {
        unit.kind == UnitKind::Hero
            && unit.relation == crate::EntityRelation::Enemy
            && unit.unit().hp <= hero.attack_damage / 2
    });
    if close_finish && matches!(action.kind(), ActionKind::AttackUnit | ActionKind::Continue) {
        return 3;
    }
    if full_inside(state) && action.kind() == ActionKind::MovePoint {
        return 1;
    }
    if hero.hp >= hero.max_hp && hero.mana >= hero.max_mana {
        return 2;
    }
    0
}

fn prepare_delay(
    state: &mut Game,
    delay: usize,
    outcome: &mut goal::Outcome,
    opponent: bool,
) -> Vec<StructuredAction> {
    let start = state.arena.tick();
    let (_, space) = state.prepare();
    let (_, landing) = trip::fountain(&space);
    let mut ledger = Vec::new();
    if delay == 0 {
        return ledger;
    }
    for _ in 0..400 {
        if full_inside(state) {
            break;
        }
        let (_, space) = state.prepare();
        let action = trip::reference(state, &space);
        trip::act(state, action, opponent, |state| {
            outcome.observe(state, start, landing)
        });
        ledger.push(action);
    }
    assert!(full_inside(state));
    assert!(delay <= 100);
    for _ in 0..delay {
        trip::act(state, StructuredAction::Continue, opponent, |state| {
            outcome.observe(state, start, landing)
        });
        ledger.push(StructuredAction::Continue);
    }
    ledger
}

pub(super) fn journey_rows(report: &mut String) -> [Vec<Row>; 8] {
    let mut groups: [Vec<Row>; 8] = Default::default();
    for setup in trip_setups4(true) {
        for delay in [0, 10, 40, 100] {
            if delay > 0 && setup.threat == trip::Threat::Finish {
                continue;
            }
            let rows = journey_episode(setup, delay, report);
            for (phase, row) in rows {
                groups[setup.side * 4 + phase].push(row);
            }
        }
    }
    writeln!(
        report,
        "journey_contexts_R_then_D_travel_depart_outbound_finish={:?}",
        groups.each_ref().map(Vec::len)
    )
    .unwrap();
    assert!(groups.iter().all(|group| !group.is_empty()));
    assert!(groups.iter().map(Vec::len).sum::<usize>() <= 2800);
    groups
}

fn journey_episode(setup: trip::Setup, delay: usize, report: &mut String) -> Vec<(usize, Row)> {
    let mut state = trip::create(setup);
    let start = state.arena.tick();
    let (_, space) = state.prepare();
    let (_, landing) = trip::fountain(&space);
    let mut outcome = goal::Outcome::new(&state);
    outcome.observe(&state, start, landing);
    let prefix = prepare_delay(
        &mut state,
        delay,
        &mut outcome,
        setup.threat != trip::Threat::None,
    );
    let mut rows = Vec::new();
    let mut last_send = 0;
    let mut ledger = Vec::new();
    for decision in 0..600 {
        if state.terminal || outcome.exact.died {
            break;
        }
        let (frame, space) = state.prepare();
        let action = data_reference(&state, &space);
        let category = phase(&state, &space, action);
        let inside = state.seats[setup.side]
            .tracker
            .own_hero()
            .unwrap()
            .pos
            .within(
                fountain_center(&state),
                Fixed::from_int(rules::FOUNTAIN_HEAL_RADIUS),
            );
        let sequence = state.seats[setup.side].sequence;
        trip::act(
            &mut state,
            action,
            setup.threat != trip::Threat::None,
            |state| outcome.observe(state, start, landing),
        );
        let sent = sequence != state.seats[setup.side].sequence;
        if sent {
            last_send = decision;
        }
        let retain =
            decision <= last_send + 3 || sent || decision % 10 == 0 || inside && category != 2;
        if retain {
            rows.push((category, Row { set: CheckedActionSet::new(frame, &space, &[action]).unwrap(),
                identity: format!("validated_functional_reference:{setup:?}:delay_decisions{delay}:tick{}:context{category}", space.tick()) }));
        }
        ledger.push(action);
        if goal::valid(setup, &outcome) {
            break;
        }
    }
    let valid = goal::valid(setup, &outcome);
    writeln!(report, "journey setup={setup:?} delay_decisions={delay} valid={valid} rows={} outcome={outcome:?} prefix={prefix:?} corrective_ledger={ledger:?}", rows.len()).unwrap();
    if valid { rows } else { vec![] }
}

fn counterfactual_rows(
    reference: &PolicyModel,
    start: &PolicyModel,
    report: &mut String,
) -> Vec<Row> {
    let mut rows = Vec::new();
    let mut counts = [0u32; 3];
    for (coverage, physics) in combat_cases4(true) {
        let mut state = game(physics);
        let (frame, space) = state.prepare();
        let alternate = start.choose(&frame, &space).unwrap().action;
        let first = reference.choose(&frame, &space).unwrap().action;
        let mut prefix = Prefix {
            physics,
            actions: vec![],
        };
        for decision in 0..60 {
            let mut state = rank::replay(&prefix);
            if state.terminal {
                break;
            }
            let (frame, space) = state.prepare();
            let action = reference.choose(&frame, &space).unwrap().action;
            if [0, 1, 2, 3, 4, 5, 10, 20, 40].contains(&decision) {
                let selected = retain_visit(&prefix, reference, &mut rows, &mut counts, report);
                assert_eq!(action, selected);
            }
            prefix.actions.push(action);
        }
        writeln!(
            report,
            "reference_ledger coverage={coverage} physics={physics:?} actions={:?}",
            prefix.actions
        )
        .unwrap();
        if alternate != first {
            prefix.actions = vec![alternate];
            for _ in 0..3 {
                if rank::replay(&prefix).terminal {
                    break;
                }
                let action = retain_visit(&prefix, reference, &mut rows, &mut counts, report);
                prefix.actions.push(action);
            }
        }
        if let Some(attack) = attack_branch(physics) {
            prefix.actions = vec![attack];
            for _ in 0..3 {
                if rank::replay(&prefix).terminal {
                    break;
                }
                let action = retain_visit(&prefix, reference, &mut rows, &mut counts, report);
                prefix.actions.push(action);
            }
        }
    }
    writeln!(
        report,
        "counterfactual_strict_supported_abstained={counts:?} rows={}",
        rows.len()
    )
    .unwrap();
    assert!(rows.len() <= 900);
    rows
}

fn retain_visit(
    prefix: &Prefix,
    reference: &PolicyModel,
    rows: &mut Vec<Row>,
    counts: &mut [u32; 3],
    report: &mut String,
) -> StructuredAction {
    let (row, action, improved, support) = supported_target(prefix, reference);
    counts[if improved {
        0
    } else if row.is_some() {
        1
    } else {
        2
    }] += 1;
    writeln!(report, "visit physics={:?} prefix={:?} reference={action:?} improved={improved} support={support:?} targets={:?}", prefix.physics, prefix.actions, row.as_ref().map(|row| row.set.actions())).unwrap();
    if let Some(row) = row {
        rows.push(row);
    }
    action
}

pub(super) fn attack_branch(physics: Physics) -> Option<StructuredAction> {
    let mut state = game(physics);
    let (_, space) = state.prepare();
    let hero = state.seats[physics.side].tracker.own_hero().unwrap();
    let has_mango = hero
        .items
        .iter()
        .flatten()
        .any(|item| item.id == ItemId(42));
    if !has_mango || hero.mana >= 75 {
        return None;
    }
    rank::candidates(&state, &space)
        .into_iter()
        .find(|action| action.kind() == ActionKind::AttackUnit)
}
