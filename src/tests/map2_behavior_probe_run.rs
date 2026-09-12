use super::*;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::io::Write as _;

const INITIAL_SHA: &str = "1739d280cb6c3fbd0df71ffe8c4a129ed3e25a0ed294b0b57931c4891c566bf4";
const WALL_LIMIT: Duration = Duration::from_secs(110);
const ARTIFACT_LIMIT: usize = 16 * 1024 * 1024;

#[test]
#[ignore = "Parameter-only initial M16 connectivity audit; zero decisions, ticks, or training."]
fn initial_m16_feature_connectivity_artifact() {
    let started = Instant::now();
    let weights =
        std::env::var("DRYSUA_MAP2_BEHAVIOR_WEIGHTS").expect("explicit initial directory");
    let weights = Path::new(&weights);
    let bytes = std::fs::read(weights.join("drysua.weights.safetensors")).expect("runtime weights");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, INITIAL_SHA);
    let model = PolicyModel::fresh_on(10091601, PolicyDevice::Cpu).expect("CPU schema carrier");
    TrainingArtifact::load_runtime_weights(&model, weights).expect("strict M16 loader");
    let parameters = model.export_parameters().expect("read-only parameters");
    let schema = model.parameter_schema().expect("named layout");
    assert_eq!(schema.len(), 62);
    let mut offset = 0;
    let mut checked = 0;
    let mut report =
        format!("sha256={digest}\nparameter_only=true\nNN_decisions=0\nengine_ticks=0\n");
    for (name, shape) in schema {
        let count = shape.iter().product::<usize>();
        let range = match name {
            "unit.0.weight" => Some(73 * 64..84 * 64),
            "trunk.0.weight" => Some(72 * 512..85 * 512),
            _ => None,
        };
        if let Some(range) = range {
            let values = &parameters[offset + range.start..offset + range.end];
            assert!(values.iter().all(|value| value.to_bits() == 0));
            checked += values.len();
            writeln!(
                report,
                "{name} shape={shape:?} inserted_input_range={range:?} positive_zero_values={}",
                values.len()
            )
            .expect("audit row");
        }
        offset += count;
    }
    assert_eq!(offset, parameters.len());
    assert_eq!(checked, 7360);
    report.push_str("CAUSAL_SCOPE: effect15 presence/stacks/timer occupy unit input73..76 and are present in the seat encoder, but their initial first-layer weights are all zero. These inputs cannot directly affect this initialized model. This is intentional initialization, not a missing wire feature or execution bug. Existing geometry/HP/action legality can still affect choices; this audit alone does not prove a specific action would change after training.\n");
    writeln!(
        report,
        "wall_seconds={:.3}",
        started.elapsed().as_secs_f64()
    )
    .expect("audit duration");
    let output = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "artifacts/temp/map2-learning-20260911/behavior/initial-m16-probe-v1/connectivity.txt",
    );
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("new connectivity audit");
    file.write_all(report.as_bytes())
        .expect("small audit write");
    assert!(report.len() < 4096);
    eprintln!("{report}");
}

#[test]
#[ignore = "Explicit bounded initial M16 behavioral artifact diagnostic; no training or qualification."]
fn initial_m16_bounded_behavior_artifact() {
    let started = Instant::now();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let weights =
        std::env::var("DRYSUA_MAP2_BEHAVIOR_WEIGHTS").expect("explicit initial weights directory");
    let weights = Path::new(&weights);
    let bytes = std::fs::read(weights.join("drysua.weights.safetensors")).expect("initial weights");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, INITIAL_SHA);
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 16);
    let model = PolicyModel::fresh_on(10091601, PolicyDevice::Cpu).expect("small CPU actor");
    TrainingArtifact::load_runtime_weights(&model, weights).expect("strict current M16 weights");
    let output = root.join("artifacts/temp/map2-learning-20260911/behavior/initial-m16-probe-v1");
    assert!(!output.exists(), "never overwrite a behavioral witness run");
    std::fs::create_dir_all(&output).expect("new behavior artifact directory");
    let mut cases: Vec<_> = FAMILIES
        .into_iter()
        .flat_map(|family| (0..2).map(move |side| fixture(family, side)))
        .collect();
    assert_eq!(cases.len(), 16);
    for _ in 0..TICK_LIMIT / 3 {
        if started.elapsed() >= WALL_LIMIT {
            break;
        }
        let active: Vec<_> = (0..cases.len())
            .filter(|index| !cases[*index].terminal)
            .collect();
        if active.is_empty() {
            break;
        }
        let prepared: Vec<_> = active
            .iter()
            .map(|index| {
                let case = &mut cases[*index];
                prepare_neural_seat_policy_sample(&mut case.seats[case.side])
                    .expect("unaltered candidate input")
            })
            .collect();
        let (frames, spaces): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
        let choices = model
            .choose_batch(&frames, &spaces)
            .expect("pure greedy autoregressive decoder");
        for ((index, choice), space) in active.into_iter().zip(choices).zip(spaces) {
            if started.elapsed() >= WALL_LIMIT {
                break;
            }
            decision(&mut cases[index], choice.action, &space, started);
        }
    }
    write_artifacts(&output, &cases, started.elapsed());
    let after = std::fs::read(weights.join("drysua.weights.safetensors"))
        .expect("unchanged initial weights");
    assert_eq!(bytes, after);
    eprintln!(
        "behavior_artifact={} wall_seconds={:.3} weights_unchanged=true",
        output.display(),
        started.elapsed().as_secs_f64()
    );
}

fn decision(case: &mut Case, action: StructuredAction, space: &ActionSpace, started: Instant) {
    assert!(space.allows(action));
    assert!(case.metrics.decisions < TICK_LIMIT / 3);
    let opportunity = opportunities(case, space);
    let proposed_wire = space.decode(action).expect("greedy action decode");
    let before = case.seats[case.side]
        .tracker
        .own_hero()
        .map(|hero| (hero.pos, hero.hp, hero.mana));
    let before_casts = case.metrics.casts;
    let before_damage = case.metrics.magic_hero_damage;
    let request = neural_policy_request_in_space(&mut case.seats[case.side], action, space)
        .expect("unchanged candidate suppression and transport")
        .1;
    let suppressed = proposed_wire.is_some() && request.is_none();
    case.metrics.decisions += 1;
    case.metrics.kinds[action.kind().index()] += 1;
    case.metrics.sent += u32::from(request.is_some());
    case.metrics.suppressed += u32::from(suppressed);
    let mut samples = Vec::with_capacity(3);
    for offset in 0..3 {
        if case.terminal || started.elapsed() >= WALL_LIMIT {
            break;
        }
        let prior_tick = case.arena.tick();
        case.advance(if offset == 0 { request } else { None });
        let events: Vec<_> = case.seats[case.side]
            .tracker
            .recent_events()
            .iter()
            .filter(|event| event.tick > prior_tick)
            .take(8)
            .collect();
        samples.push(format!("{events:?}"));
    }
    if case.trace.len() < TRACE_LIMIT
        && (case.metrics.decisions <= 8
            || case.metrics.decisions.is_multiple_of(2)
            || case.metrics.casts > before_casts)
    {
        let hero = case.seats[case.side]
            .tracker
            .own_hero()
            .map(|hero| (hero.pos, hero.hp, hero.mana));
        case.trace.push(format!("tick={} action={action:?} decoded={proposed_wire:?} suppressed={suppressed} wire={request:?} before={before:?} after={hero:?} confirmed_casts={} magic_hero_damage={} max_stacks={} opportunity={opportunity} rejection={:?} seat_event_samples={samples:?}", space.tick(), case.metrics.casts - before_casts, case.metrics.magic_hero_damage - before_damage, case.metrics.max_stacks, case.seats[case.side].last_rejection));
    }
}

fn opportunities(case: &mut Case, space: &ActionSpace) -> String {
    assert_eq!(case.view().tick, space.tick());
    assert!(case.side < 2);
    let Some(hero) = case.seats[case.side].tracker.own_hero() else {
        return "dead".into();
    };
    let enemy = case
        .view()
        .units
        .iter()
        .find(|unit| unit.kind == UnitKind::Hero && unit.team != hero.team);
    let hittable = hittable_raze_slots(hero, enemy, space);
    let enemy_stacks = enemy.map(raze_stacks);
    case.metrics.raze_opportunities += u32::from(!hittable.is_empty());
    case.metrics.third_stack_opportunities +=
        u32::from(!hittable.is_empty() && enemy_stacks.is_some_and(|stacks| stacks >= 2));
    let buy = mango_buy(space);
    let using = space.allows(mango_use());
    case.metrics.mango_buy_opportunities += u32::from(buy.is_some());
    case.metrics.mango_use_opportunities += u32::from(using);
    format!(
        "seat_geometry_hittable_slots={hittable:?},enemy_stacks={enemy_stacks:?},mango_buy={buy:?},mango_use={using},home_move={:?},stash_mango={}",
        goal_move(space, case.goal),
        case.stash_mango()
    )
}

fn hittable_raze_slots(hero: &UnitView, enemy: Option<&UnitView>, space: &ActionSpace) -> Vec<u8> {
    let Some(enemy) = enemy else {
        return Vec::new();
    };
    assert_ne!(hero.team, enemy.team);
    assert_eq!(hero.kind, UnitKind::Hero);
    (0..3)
        .filter(|slot| {
            let at = bota_server::game::point_along(
                hero.pos,
                hero.pos + bota_server::game::heading_of(hero.facing),
                Fixed::from_int(rules::RAZE_DISTANCE[*slot as usize]),
            );
            space.allows(cast(ControlledUnit::Hero, *slot))
                && enemy.pos.within(at, Fixed::from_int(rules::RAZE_RADIUS))
        })
        .collect()
}

fn write_artifacts(output: &Path, cases: &[Case], elapsed: Duration) {
    assert!(cases.len() <= 16);
    let mut report = String::from(
        "family\tside\tticks\tdecisions\tcasts\thero_hit_casts\tmagic_hero_damage\tno_damage_unknown\tvisible_enemy_no_damage\tmax_stacks\tstack_increases\traze_opportunities\tthird_stack_opportunities\tmana_spent\tmango_bought\tmango_arrivals\tmango_used\tmango_held\tmango_stash\tmanual_mana\tcreep_damage\treward_creep_damage\tlane_hits\tneutral_hits\tunknown_incoming\thome_progress\tpath\treversals\taway_ticks\tstationary_ticks\tsent\tsuppressed\trejections\n",
    );
    let mut trace = format!(
        "model=M16 sha256={INITIAL_SHA} seed_fixture=10091600 seed_fresh_model=10091601 device=CPU greedy=true teacher=false custom_safety_masks=false learning=false elapsed_seconds={:.3}\n",
        elapsed.as_secs_f64()
    );
    trace.push_str("SETUP_GROUND_TRUTH_ONLY: native Map2 geometry/buildings retained; fixture starts at tick1201 (neutral tick2700 after native fill_camps); level3 heroes, razes level1; passive opponent Stand/no requests. Chain enemy 450 ahead with same-caster stacks2/timer240; empty opponent at native fountain. Mango buy/delivery hero 1100x1100 from native fountain, buy gold65/delivery stash1. Recovery hp100/mana0, 180x180 lane-side of native mid melee barracks; exogenous home goal is measurement ONLY, never tensor input. Lane exposure three native melee units 100 away without a march route (isolated idle encounter, not a full wave). Neutral exposure 160 from actual small camps2/22 after native fill. No world state is read by policy or outcome metrics.\n");
    trace.push_str("ATTRIBUTION: cast confirmation requires AbilityCast plus cooldown increase or mana drop>=70 on same hero generation; Learn-only events excluded. No observed magic damage is UNKNOWN for full fog attribution, even in empty fixture; visible_enemy_no_damage is only a visible-target failure, not proof of no unseen hits. Incoming damage counted from full native event batch; lane/neutral categories use current/prior seat-visible identities. Exogenous home progress is Euclidean delta, path is per-tick integer-coordinate length, reversals negative displacement dot product, stationary zero integer displacement. Controls are separate deterministic tests, never NN success. Terminal cases stop; partial time-budget cases retained; no win claims.\n");
    for case in cases {
        let metric = &case.metrics;
        let held = case.seats[case.side]
            .tracker
            .own_hero()
            .map(|hero| mango_count(&hero.items))
            .unwrap_or(0);
        writeln!(report, "{:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{}\t{}\t{}\t{}\t{}\t{}", case.family, case.side, metric.ticks, metric.decisions, metric.casts, metric.hero_hit_casts, metric.magic_hero_damage, metric.no_damage_unknown, metric.visible_enemy_no_damage, metric.max_stacks, metric.stack_increases, metric.raze_opportunities, metric.third_stack_opportunities, metric.mana_spent, metric.mango_bought, metric.mango_arrivals, metric.mango_used, held, case.stash_mango(), metric.manual_mana, metric.creep_damage, metric.reward_creep_damage, metric.lane_hits, metric.neutral_hits, metric.unknown_incoming, metric.progress(), metric.path, metric.reversals, metric.away_ticks, metric.stationary_ticks, metric.sent, metric.suppressed, case.seats[case.side].rejections).expect("summary formatting");
        writeln!(
            trace,
            "CASE {:?} side={} terminal={} complete={} metrics={metric:?}",
            case.family,
            case.side,
            case.terminal,
            metric.ticks == TICK_LIMIT
        )
        .expect("case formatting");
        assert!(case.trace.len() <= TRACE_LIMIT);
        for row in &case.trace {
            writeln!(trace, "{row}").expect("trace formatting");
        }
    }
    assert!(report.len() + trace.len() < ARTIFACT_LIMIT);
    for (name, content) in [
        ("summary.tsv", report.as_bytes()),
        ("witnesses.txt", trace.as_bytes()),
    ] {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join(name))
            .expect("new artifact only");
        file.write_all(content).expect("bounded artifact write");
    }
    eprintln!("{report}");
}
