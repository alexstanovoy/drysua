#![allow(
    clippy::float_arithmetic,
    reason = "behavioral metrics and optimizer configuration"
)]

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{
    ActionKind, ActionSpace, ActiveOrderUpdate, ActivePolicyOrder, AdamConfig, Arena, ArenaConfig,
    BehavioralTrainer, FeatureEncoder, FeatureFrame, IMITATION_RULES_AUDIT_VERSION, ImitationPool,
    ImitationSample, ItemReadiness, LocalPolicyState, MAX_IMITATION_SAMPLES, OfflineEvaluation,
    OrderPersistence, PolicyDevice, PolicyModel, PpoRng, Request, SampleIdentity, SeedNamespace,
    SeedNamespaces, StateTracker, StructuredAction, Teacher, TeacherCoverage, TrainingArtifact,
    TrainingScope, active_order_update_for_sent,
};
use bota_proto::{MapId, ServerMsg, SlotId, Team};
use sha2::{Digest, Sha256};

use crate::persistence::training::PolicyOrderBookkeeping;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const CAPACITIES: [usize; 3] = [16384, 6144, 6144];
const DAGGER_CAPACITY: usize = 4096;
const _: () = assert!(
    CAPACITIES[0] + CAPACITIES[1] + CAPACITIES[2] + DAGGER_CAPACITY == MAX_IMITATION_SAMPLES
);

/// Fresh Map0-only BC probe. All outputs are diagnostic, never automatically promoted.
pub fn run_neural_training(config: &NeuralTrainingConfig, output: &Path) -> Result<()> {
    validate_config(config)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/temp")
        .canonicalize()?;
    if !output
        .parent()
        .ok_or("output needs an existing parent")?
        .canonicalize()?
        .starts_with(root)
    {
        return Err("neural training output must be below artifacts/temp".into());
    }
    fs::create_dir(output)?;
    let deadline = Instant::now() + config.wall_time;
    fs::write(
        output.join("status.json"),
        "{\"accepted\":false,\"phase\":\"collecting\"}\n",
    )?;
    let mut data = collect_dataset(config, Some(deadline))?;
    write_dataset(&data, output)?;
    let (checkpoint, actor) = train_checkpoints(config, output, deadline, &mut data)?;
    fs::write(
        output.join("selected-diagnostic.txt"),
        format!("{}\n", checkpoint.display()),
    )?;
    TrainingArtifact::load_runtime_weights(&actor, &checkpoint)?;
    write_branch_accuracy(
        &actor,
        &data,
        SeedNamespace::Promotion,
        &output.join("final-branches.txt"),
    )?;
    let held_out =
        OfflineEvaluation::evaluate_held_out(&actor, &data.pool, data.coverage[1].clone())?;
    fs::write(
        output.join("held-out.txt"),
        format!(
            "stratified=true population_accuracy=false\n{:?}\nnoncontinue_full={:?}\nreal_continue_baseline={}\n",
            held_out.metrics(),
            noncontinue_full(held_out.metrics()),
            real_continue(&data, 2)
        ),
    )?;
    let final_games = evaluate_pair(&actor, config.seed + 200, config.tick_limit, Some(deadline))?;
    fs::write(
        output.join("final-gameplay.txt"),
        format!("{final_games:?}\n"),
    )?;
    let candidate = output.join("ppo-candidate");
    fs::create_dir(&candidate)?;
    TrainingArtifact::save_runtime_weights(&actor, &candidate)?;
    fs::write(
        output.join("status.json"),
        "{\"accepted\":false,\"phase\":\"completed_diagnostic\",\"promotion\":\"not_performed\"}\n",
    )?;
    println!(
        "held_out_stratified_kind={:?} held_out_stratified_full={:?} final_rank={:?} accepted=false",
        held_out.metrics().overall.kind_agreement(),
        held_out.metrics().overall.full_agreement(),
        gameplay_rank(&final_games)
    );
    Ok(())
}

fn train_checkpoints(
    config: &NeuralTrainingConfig,
    output: &Path,
    deadline: Instant,
    data: &mut NeuralDataset,
) -> Result<(PathBuf, PolicyModel)> {
    let mut session = TrainingSession::new(config, data)?;
    session.baseline(config, output, deadline, data)?;
    session.stage(config, output, deadline, data, 0)?;
    for round in 0..config.dagger_rounds {
        if Instant::now() + Duration::from_secs(420) >= deadline {
            fs::write(
                output.join("dagger-deadline.txt"),
                format!("round {round} not scheduled: insufficient complete-match budget\n"),
            )?;
            return Err(
                "required DAgger pass cannot fit before deadline; experiment unaccepted".into(),
            );
        }
        session
            .actor
            .import_parameters(&session.model.export_parameters()?)?;
        let directory = output.join(format!("dagger-data-{}", round + 1));
        fs::create_dir(&directory)?;
        match append_dagger(
            config,
            data,
            &session.actor,
            round,
            Some(deadline),
            &directory,
        ) {
            Ok(()) => {
                session.trainer.rebind_pool(&data.pool)?;
                session.stage(config, output, deadline, data, round + 1)?;
            }
            Err(error) => {
                fs::write(
                    directory.join("collection-error.txt"),
                    format!("{error}\nno_partial_pool_append=true\n"),
                )?;
                println!("dagger round={} failed_diagnostic={error}", round + 1);
                return Err(error);
            }
        }
    }
    let (_, checkpoint) = session
        .best
        .ok_or("no complete selection cohort before deadline")?;
    Ok((checkpoint, session.actor))
}

type SelectionRank = (i64, i64, i64, i64, i64);

struct TrainingSession {
    model: PolicyModel,
    actor: PolicyModel,
    trainer: BehavioralTrainer,
    best: Option<(SelectionRank, PathBuf)>,
    initialization: Initialization,
}

#[derive(Debug)]
enum Initialization {
    Fresh,
    CurrentWeights { source: PathBuf, sha256: String },
    SelectedM10 { source: PathBuf, sha256: String },
}

impl Initialization {
    fn provenance(&self) -> String {
        let (kind, source, sha256, old_tuple) = match self {
            Self::Fresh => ("Fresh", None, "none", "none"),
            Self::CurrentWeights { source, sha256 } => {
                ("CurrentWeights", Some(source), sha256.as_str(), "current")
            }
            Self::SelectedM10 { source, sha256 } => (
                "SelectedM10",
                Some(source),
                sha256.as_str(),
                "A3:1755359086494840931,F9:9669721049329356661,M10:720439888929233033,PPO18:6877503070358232325,audit15",
            ),
        };
        format!(
            "initialization={kind}\nsource={source:?}\nsource_sha256={sha256}\nold_tuple={old_tuple}\ncurrent_model={}:{}\ncurrent_feature={}:{}\noptimizer=new_Adam_no_resume\nlearning_rate=0.001 beta1=0.9 beta2=0.999 epsilon=1e-8 gradient_clip=0.5\nteacher_overrides=0\n",
            crate::MODEL_SCHEMA_VERSION,
            crate::MODEL_SCHEMA_HASH,
            crate::FEATURE_SCHEMA_VERSION,
            crate::FEATURE_SCHEMA_HASH
        )
    }
}

impl TrainingSession {
    fn baseline(
        &mut self,
        config: &NeuralTrainingConfig,
        output: &Path,
        deadline: Instant,
        data: &NeuralDataset,
    ) -> Result<()> {
        let baseline = output.join("baseline");
        fs::create_dir(&baseline)?;
        TrainingArtifact::save_runtime_weights(&self.model, &baseline)?;
        fs::write(
            baseline.join("initialization.txt"),
            self.initialization.provenance(),
        )?;
        let validation = OfflineEvaluation::evaluate_validation(
            &self.model,
            &data.pool,
            data.coverage[0].clone(),
        )?;
        fs::write(
            baseline.join("offline.txt"),
            format!(
                "{:?}\nnoncontinue_full={:?}\n",
                validation.metrics(),
                noncontinue_full(validation.metrics())
            ),
        )?;
        write_branch_accuracy(
            &self.model,
            data,
            SeedNamespace::Validation,
            &baseline.join("selection-branches.txt"),
        )?;
        self.select_checkpoint(config, baseline, validation.metrics(), deadline)
    }

    fn new(config: &NeuralTrainingConfig, data: &NeuralDataset) -> Result<Self> {
        let (model, initialization) = initialize_model(config)?;
        let trainer = BehavioralTrainer::new(
            64,
            config.seed ^ 0x4243,
            AdamConfig {
                learning_rate: 1.0e-3,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1.0e-8,
                gradient_clip: 0.5,
            },
            &model,
            &data.pool,
        )?;
        Ok(Self {
            model,
            actor: PolicyModel::fresh(config.seed)?,
            trainer,
            best: None,
            initialization,
        })
    }

    fn stage(
        &mut self,
        config: &NeuralTrainingConfig,
        output: &Path,
        deadline: Instant,
        data: &NeuralDataset,
        stage: usize,
    ) -> Result<()> {
        let epochs = if stage == 0 {
            config.epochs
        } else {
            config.dagger_epochs
        };
        for epoch in 1..=epochs {
            if Instant::now() + Duration::from_secs(180) >= deadline {
                return Err(
                    "requested training epochs incomplete before deadline; experiment unaccepted"
                        .into(),
                );
            }
            let started = Instant::now();
            let trained = self.trainer.train_epoch(&self.model, &data.pool)?;
            let checkpoint = self.save_epoch(config, output, stage, epoch)?;
            let validation = OfflineEvaluation::evaluate_validation(
                &self.model,
                &data.pool,
                data.coverage[0].clone(),
            )?;
            let metrics = validation.metrics();
            fs::write(
                checkpoint.join("offline.txt"),
                format!(
                    "stratified=true population_accuracy=false\nloss={}\n{:?}\nnoncontinue_full={:?}\n",
                    trained.average_loss,
                    metrics,
                    noncontinue_full(metrics)
                ),
            )?;
            println!(
                "stage={stage} epoch={epoch} updates={} loss={:.6} stratified_kind={:?} stratified_full={:?} noncontinue_full={:?} train_seconds={:.3}",
                self.trainer.counters().global_update,
                trained.average_loss,
                metrics.overall.kind_agreement(),
                metrics.overall.full_agreement(),
                noncontinue_full(metrics),
                started.elapsed().as_secs_f64()
            );
            fs::write(
                checkpoint.join("sampling-baseline.txt"),
                format!(
                    "real_selection_continue_baseline={}\nstratified_continue_baseline={:?}\n",
                    real_continue(data, 1),
                    metrics.overall.continue_ratio()
                ),
            )?;
            if epoch == epochs || (stage == 0 && matches!(epoch, 8 | 16 | 32)) {
                write_branch_accuracy(
                    &self.model,
                    data,
                    SeedNamespace::Validation,
                    &checkpoint.join("selection-branches.txt"),
                )?;
                self.select_checkpoint(config, checkpoint, metrics, deadline)?;
            }
            std::io::stdout().flush()?;
        }
        Ok(())
    }

    fn save_epoch(
        &self,
        config: &NeuralTrainingConfig,
        output: &Path,
        stage: usize,
        epoch: u32,
    ) -> Result<PathBuf> {
        assert!(stage <= config.dagger_rounds);
        assert!((1..=64).contains(&epoch));
        let checkpoint = output.join(format!("stage-{stage}-epoch-{epoch:03}"));
        fs::create_dir(&checkpoint)?;
        TrainingArtifact::save_runtime_weights(&self.model, &checkpoint)?;
        fs::write(
            checkpoint.join("initialization.txt"),
            format!(
                "{}seed={} epoch={epoch} stage={stage} updates={}\n",
                self.initialization.provenance(),
                config.seed,
                self.trainer.counters().global_update
            ),
        )?;
        Ok(checkpoint)
    }

    fn select_checkpoint(
        &mut self,
        config: &NeuralTrainingConfig,
        checkpoint: PathBuf,
        metrics: &OfflineEvaluation,
        deadline: Instant,
    ) -> Result<()> {
        self.actor
            .import_parameters(&self.model.export_parameters()?)?;
        let games = match evaluate_pair(
            &self.actor,
            config.seed + 100,
            config.tick_limit,
            Some(deadline),
        ) {
            Ok(games) => games,
            Err(error) => {
                fs::write(
                    checkpoint.join("selection-error.txt"),
                    format!("{error}\nnot_a_win=true\n"),
                )?;
                println!(
                    "selection checkpoint={} failed_diagnostic={error}",
                    checkpoint.display()
                );
                return Ok(());
            }
        };
        let rank = selection_rank(gameplay_rank(&games), metrics);
        fs::write(
            checkpoint.join("selection.txt"),
            format!("{games:?}\nrank={rank:?}\n"),
        )?;
        println!(
            "selection checkpoint={} rank={rank:?} candidate_orders={:?} left_fountain={:?}",
            checkpoint.display(),
            games
                .iter()
                .map(|game| game.orders[game.candidate])
                .collect::<Vec<_>>(),
            games
                .iter()
                .map(|game| game.left_fountain_area[game.candidate])
                .collect::<Vec<_>>()
        );
        if self
            .best
            .as_ref()
            .is_none_or(|(previous, _)| rank > *previous)
        {
            self.best = Some((rank, checkpoint));
        }
        Ok(())
    }
}

fn append_dagger(
    config: &NeuralTrainingConfig,
    data: &mut NeuralDataset,
    model: &PolicyModel,
    round: usize,
    deadline: Option<Instant>,
    output: &Path,
) -> Result<()> {
    assert!(round < config.dagger_rounds);
    assert!(config.dagger_rounds <= 2);
    let seed = config.seed + 50 + round as u64;
    let mut reservoir = Reservoir::dagger(DAGGER_CAPACITY / config.dagger_rounds, seed ^ 0xda66e2);
    TrainingArtifact::save_runtime_weights(model, output)?;
    let mut games = Vec::with_capacity(2);
    for side in 0..2 {
        games.push(run_game(
            Some(model),
            seed,
            side,
            config.tick_limit,
            deadline,
            Some((&mut reservoir, SeedNamespace::Training)),
        )?);
    }
    let raw = reservoir.counts;
    let raw_by_side = reservoir.counts_by_side;
    for (key, (seen, _)) in &reservoir.selector.counts {
        *data.seen_branches[0].entry(*key).or_default() += seen;
    }
    let samples = reservoir.into_samples()?;
    assert!(data.pool.len() + samples.len() <= MAX_IMITATION_SAMPLES);
    let count = samples.len();
    for sample in samples {
        assert_eq!(sample.source(), crate::ImitationSource::Dagger);
        assert_eq!(sample.identity().namespace(), SeedNamespace::Training);
        assert!(data.pool.push(sample)?.is_none());
    }
    for (phase, counts) in raw.iter().enumerate() {
        for (kind, count) in counts.iter().enumerate() {
            data.seen[0][phase][kind] += count;
        }
    }
    for (side, phases) in raw_by_side.iter().enumerate() {
        for (phase, counts) in phases.iter().enumerate() {
            for (kind, count) in counts.iter().enumerate() {
                data.seen_by_side[0][side][phase][kind] += count;
            }
        }
    }
    fs::write(
        output.join("collection.txt"),
        format!(
            "learner_states_only=true fake_teacher_sends=false\nretained={count}\nseen={raw:?}\ngames={games:?}\n"
        ),
    )?;
    write_dataset(data, output)?;
    println!(
        "dagger round={} seed={seed} retained={count} pool_samples={}",
        round + 1,
        data.pool.len()
    );
    Ok(())
}

fn noncontinue_full(metrics: &OfflineEvaluation) -> Option<f64> {
    let (matching, total) = metrics.overall.families[1..]
        .iter()
        .fold((0usize, 0usize), |(matching, total), entry| {
            (matching + entry.matching, total + entry.total)
        });
    (total > 0).then(|| matching as f64 / total as f64)
}

fn selection_rank(gameplay: (i64, i64, i64, i64), metrics: &OfflineEvaluation) -> SelectionRank {
    let score = noncontinue_full(metrics).map_or(-1, |score| (score * 1000000.0) as i64);
    (gameplay.0, gameplay.1, gameplay.2, gameplay.3, score)
}

#[derive(Clone, Debug)]
pub struct NeuralTrainingConfig {
    pub seed: u64,
    pub training_games: usize,
    pub epochs: u32,
    pub tick_limit: u32,
    pub wall_time: Duration,
    pub device: PolicyDevice,
    pub dagger_rounds: usize,
    pub dagger_epochs: u32,
    pub initial_weights: Option<PathBuf>,
    pub initialize_selected_m10: Option<PathBuf>,
}

impl Default for NeuralTrainingConfig {
    fn default() -> Self {
        Self {
            seed: 9840000,
            training_games: 6,
            epochs: 64,
            tick_limit: 108900,
            wall_time: Duration::from_secs(1800),
            device: PolicyDevice::Cpu,
            dagger_rounds: 1,
            dagger_epochs: 32,
            initial_weights: None,
            initialize_selected_m10: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NeuralGameReport {
    pub seed: u64,
    pub candidate: usize,
    pub ticks: u32,
    pub winner: Option<Team>,
    pub decisions: [u32; 2],
    pub orders: [u32; 2],
    pub order_fingerprints: [u64; 2],
    pub rejections: [u32; 2],
    pub kinds: [[u64; ActionKind::COUNT]; 2],
    pub progress: Option<(i64, i64)>,
    pub opening_kinds: [[u64; ActionKind::COUNT]; 2],
    pub left_fountain_area: [bool; 2],
}

struct NeuralSeat {
    tracker: StateTracker,
    encoder: FeatureEncoder,
    local: LocalPolicyState,
    persistence: OrderPersistence,
    order_bookkeeping: PolicyOrderBookkeeping,
    readiness: ItemReadiness,
    pending: Option<(u32, Option<ActivePolicyOrder>)>,
    decisions: u32,
    sequence: u32,
    orders_hash: DefaultHasher,
    rejections: u32,
    kinds: [u64; ActionKind::COUNT],
    opening_kinds: [u64; ActionKind::COUNT],
    fountain: Option<bota_proto::Vec2>,
    left_fountain_area: bool,
}

#[cfg(test)]
pub(crate) struct NeuralSeatOrderContractProbe(NeuralSeat);

#[cfg(test)]
impl NeuralSeatOrderContractProbe {
    pub(crate) fn new(side: usize, messages: &[ServerMsg]) -> Self {
        Self(NeuralSeat::new(side as u8, messages).expect("NeuralSeat order-contract seat"))
    }

    pub(crate) fn observe(&mut self, messages: &[ServerMsg]) {
        assert!(
            self.0
                .observe(messages)
                .expect("NeuralSeat probe observations")
                .is_none()
        );
    }

    pub(crate) fn decide(
        &mut self,
        model: &PolicyModel,
        candidate: bool,
    ) -> (FeatureFrame, Option<Request>, Option<ActivePolicyOrder>) {
        let (action, space) = if candidate {
            self.0
                .order_bookkeeping
                .enable_candidate(&self.0.persistence)
                .expect("candidate role");
            self.0
                .neural_choice(model)
                .expect("NeuralSeat probe choice")
        } else {
            let space =
                ActionSpace::from_tracker_with_readiness(&self.0.tracker, &self.0.readiness)
                    .expect("legacy probe space");
            let frame = self.0.frame(&space).expect("legacy probe input");
            (
                model
                    .choose(&frame, &space)
                    .expect("legacy probe choice")
                    .action,
                space,
            )
        };
        let frame = self.0.frame(&space).expect("NeuralSeat probe input");
        let request = self
            .0
            .issue(action, &space)
            .expect("NeuralSeat probe send")
            .map(|issued| Request {
                seq: self.0.sequence,
                unit: issued.unit,
                order: issued.order,
            });
        (frame, request, self.0.local.active_order())
    }

    pub(crate) fn legacy_persistence(&self) -> OrderPersistence {
        self.0.persistence
    }
}

impl NeuralSeat {
    fn new(slot: u8, messages: &[ServerMsg]) -> Result<Self> {
        let ServerMsg::MatchStart { info } = &messages[0] else {
            return Err("initial MatchStart missing".into());
        };
        assert_eq!(info.map, MapId(0));
        assert_eq!(info.picks.len(), 2);
        let tracker = StateTracker::new(SlotId(slot), info)?;
        let mut seat = Self {
            encoder: FeatureEncoder::new(&tracker),
            tracker,
            local: LocalPolicyState::new(0),
            persistence: OrderPersistence::default(),
            order_bookkeeping: PolicyOrderBookkeeping::Legacy,
            readiness: ItemReadiness::new(),
            pending: None,
            decisions: 0,
            sequence: 0,
            orders_hash: DefaultHasher::new(),
            rejections: 0,
            kinds: [0; ActionKind::COUNT],
            opening_kinds: [0; ActionKind::COUNT],
            fountain: None,
            left_fountain_area: false,
        };
        seat.observe(messages)?;
        Ok(seat)
    }

    fn observe(&mut self, messages: &[ServerMsg]) -> Result<Option<Team>> {
        assert!(messages.len() <= 4);
        assert!(!messages.is_empty());
        let mut winner = None;
        for message in messages {
            match message {
                ServerMsg::Snapshot { view } => {
                    let previous = self.tracker.own_hero().map(|hero| hero.id);
                    self.tracker.observe_snapshot(view)?;
                    self.observe_opening(view);
                    if previous != self.tracker.own_hero().map(|hero| hero.id) {
                        self.persistence.clear_body_for(None);
                        self.order_bookkeeping.clear_body_for(None);
                        self.local.set_active_order(view.tick, None)?;
                        self.pending = None;
                    }
                }
                ServerMsg::Events { tick, events } => {
                    self.tracker.observe_events(*tick, events)?;
                    self.order_bookkeeping.reconcile(
                        &self.tracker,
                        &mut self.local,
                        &mut self.pending,
                    )?;
                    self.encoder.observe(&self.tracker)?;
                }
                ServerMsg::OrderRejected { seq, .. } => {
                    self.rejections += 1;
                    self.persistence.observe_rejection(*seq);
                    self.order_bookkeeping.observe_rejection(*seq);
                    self.readiness.note_rejected(*seq);
                    if let Some((pending, previous)) = self.pending
                        && pending == *seq
                    {
                        self.local.restore_active_order(
                            self.tracker.current().ok_or("snapshot missing")?.tick,
                            previous,
                        )?;
                        self.pending = None;
                    }
                }
                ServerMsg::MatchOver {
                    winner: team,
                    stats,
                } => {
                    assert_eq!(stats.slots.len(), 2);
                    assert_eq!(
                        stats.duration,
                        self.tracker.current().ok_or("snapshot missing")?.tick
                    );
                    winner = Some(*team);
                }
                ServerMsg::MatchStart { .. } => {}
                _ => return Err("unexpected builtin seat message".into()),
            }
        }
        Ok(winner)
    }

    fn frame(&mut self, space: &ActionSpace) -> Result<FeatureFrame> {
        let mut frame = FeatureFrame::new();
        self.encoder.encode(
            &self.tracker,
            space,
            &self.readiness,
            &self.local,
            &mut frame,
        )?;
        assert!(frame.matches_action_space(space));
        assert!(frame.is_finite());
        Ok(frame)
    }

    fn observe_opening(&mut self, view: &bota_proto::WorldView) {
        if self.fountain.is_none() {
            self.fountain = view
                .units
                .iter()
                .find(|unit| {
                    unit.team == self.tracker.team() && unit.kind == bota_proto::UnitKind::Fountain
                })
                .map(|unit| unit.pos);
        }
        if view.tick <= 3000
            && let Some(origin) = self.fountain
            && let Some(hero) = self.tracker.own_hero()
        {
            self.left_fountain_area |= !hero.pos.within(origin, bota_proto::Fixed::from_int(1200));
        }
    }

    /// Predicts without opting a Teacher transport into candidate deduplication.
    fn neural_choice(&mut self, model: &PolicyModel) -> Result<(StructuredAction, ActionSpace)> {
        self.order_bookkeeping.enable_observer(&self.persistence)?;
        self.order_bookkeeping
            .reconcile(&self.tracker, &mut self.local, &mut self.pending)?;
        let space = ActionSpace::from_tracker_with_readiness(&self.tracker, &self.readiness)?;
        let frame = self.frame(&space)?;
        let action = model.choose(&frame, &space)?.action;
        assert!(space.allows(action));
        Ok((action, space))
    }

    fn issue(
        &mut self,
        action: StructuredAction,
        space: &ActionSpace,
    ) -> Result<Option<crate::IssuedOrder>> {
        assert!(self.decisions < 36300);
        self.decisions += 1;
        self.kinds[action.kind().index()] += 1;
        if space.tick() <= 3000 {
            self.opening_kinds[action.kind().index()] += 1;
        }
        self.local.note_decision(space.tick(), action.kind())?;
        let persistence = self.order_bookkeeping.transport(&self.persistence);
        let Some(issued) = persistence.should_send(space.decode(action)?) else {
            return Ok(None);
        };
        let previous = self.local.active_order();
        self.sequence += 1;
        (space.tick(), issued.unit, issued.order).hash(&mut self.orders_hash);
        let preserves = self.order_bookkeeping.record_sent(
            &mut self.persistence,
            self.sequence,
            issued,
            &self.tracker,
        )?;
        self.readiness.note_sent(self.sequence, issued, space);
        let update = if preserves {
            ActiveOrderUpdate::Preserve
        } else {
            active_order_update_for_sent(
                self.order_bookkeeping.effective(&self.persistence),
                issued.unit,
                self.sequence,
                action.kind(),
            )
        };
        match update {
            ActiveOrderUpdate::Preserve => {}
            ActiveOrderUpdate::Replace(None) if previous.is_none() => self.pending = None,
            ActiveOrderUpdate::Replace(next) => {
                if let Some(kind) = next {
                    self.local
                        .set_active_order_from_issued(space.tick(), kind, issued)?;
                } else {
                    self.local.set_active_order(space.tick(), None)?;
                }
                self.pending = Some((self.sequence, previous));
            }
        }
        Ok(Some(issued))
    }
}

struct CoverageSelector {
    capacity: usize,
    seen: u64,
    rng: PpoRng,
    cells: Vec<usize>,
    kept: [usize; 10],
    opening_priority: bool,
}

impl CoverageSelector {
    fn new(capacity: usize, seed: u64, opening_priority: bool) -> Self {
        assert!(capacity <= MAX_IMITATION_SAMPLES);
        Self {
            capacity,
            seen: 0,
            rng: PpoRng::new(seed),
            cells: Vec::with_capacity(capacity),
            kept: [0; 10],
            opening_priority,
        }
    }
    fn minimum(&self, cell: usize) -> usize {
        assert!(cell < 10);
        if self.opening_priority && cell < 2 {
            (self.capacity / 4).min(64)
        } else if self.opening_priority {
            (self.capacity / 16).min(4)
        } else {
            (self.capacity / 10).min(4)
        }
    }
    fn destination(&mut self, cell: usize) -> Result<Option<usize>> {
        assert!(cell < 10);
        assert!(self.seen < 108900 * 2 * 10);
        self.seen += 1;
        if self.capacity == 0 {
            return Ok(None);
        }
        let index = if self.cells.len() < self.capacity {
            self.cells.len()
        } else if self.kept[cell] < self.minimum(cell) {
            let start = self.rng.next_u64()? as usize % self.capacity;
            (0..self.capacity)
                .map(|offset| (start + offset) % self.capacity)
                .find(|index| self.kept[self.cells[*index]] > self.minimum(self.cells[*index]))
                .ok_or("no unprotected stratum available")?
        } else {
            let index = (self.rng.next_u64()? % self.seen) as usize;
            if index >= self.capacity {
                return Ok(None);
            }
            let previous = self.cells[index];
            if previous != cell && self.kept[previous] <= self.minimum(previous) {
                return Ok(None);
            }
            index
        };
        if index == self.cells.len() {
            self.cells.push(cell);
        } else {
            self.kept[self.cells[index]] -= 1;
            self.cells[index] = cell;
        }
        self.kept[cell] += 1;
        assert_eq!(self.kept.iter().sum::<usize>(), self.cells.len());
        Ok(Some(index))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BranchKey {
    kind: usize,
    body: usize,
    target: usize,
    slot: usize,
    point: usize,
    cell: usize,
}

impl BranchKey {
    fn new(action: StructuredAction, cell: usize) -> (Self, crate::ActionTarget) {
        use crate::{ActionTarget as Target, PutPointTarget};
        use StructuredAction::*;
        let (slot, target) = match action {
            MovePoint { point, .. } | AttackMovePoint { point, .. } => (0, Target::Point(point)),
            FollowUnit { target, .. } | AttackUnit { target, .. } => (0, Target::Entity(target)),
            Cast { slot, target, .. } => (usize::from(slot.0) + 1, target),
            Use { slot, target, .. } => (usize::from(slot.0) + 1, target),
            PutUnit { source, target, .. } => (usize::from(source.0) + 1, Target::Entity(target)),
            PutPoint { source, target, .. } => (
                usize::from(source.0) + 1,
                match target {
                    PutPointTarget::Underfoot => Target::None,
                    PutPointTarget::Point(point) => Target::Point(point),
                },
            ),
            Sell { slot, .. } => (usize::from(slot.0) + 1, Target::None),
            Swap { from, to, .. } => (
                1 + usize::from(from.0) * 15 + usize::from(to.0),
                Target::None,
            ),
            Learn { slot } => (usize::from(slot.0) + 1, Target::None),
            Buy { item, .. } => (item.0 + 1, Target::None),
            Continue | Stop { .. } | Hold { .. } | Take { .. } => (0, Target::None),
        };
        assert!(slot < 256);
        assert!(cell < 10);
        (
            Self {
                kind: action.kind().index(),
                body: action.controlled_unit().map_or(2, |unit| unit.index()),
                target: 0,
                slot,
                point: 0,
                cell,
            },
            target,
        )
    }

    fn from_space(action: StructuredAction, space: &ActionSpace, cell: usize) -> Self {
        let (mut key, target) = Self::new(action, cell);
        match target {
            crate::ActionTarget::None => {}
            crate::ActionTarget::Entity(index) => {
                let candidate = &space.entity_candidates()[index.0];
                let relation = match candidate.relation {
                    crate::EntityRelation::Own => 0,
                    crate::EntityRelation::Allied => 1,
                    crate::EntityRelation::Enemy => 2,
                    crate::EntityRelation::Neutral => 3,
                };
                key.target = 1 + candidate.kind as usize * 4 + relation;
            }
            crate::ActionTarget::Point(index) => {
                use crate::PointSource;
                key.point = match space.point_candidates()[index.0].source {
                    PointSource::Tactical { .. } => 1,
                    PointSource::StaticTree => 2,
                    PointSource::PlantedTree => 3,
                    PointSource::BuildingLanding(_) => 4,
                    PointSource::Fountain(_) => 5,
                    PointSource::Tower(_) => 6,
                    PointSource::PredictedHero(_) => 7,
                    PointSource::PredictedCreep(_) => 8,
                };
            }
        }
        assert!(key.target <= 48);
        assert!(key.point <= 8);
        key
    }

    fn from_sample(sample: &ImitationSample) -> Self {
        let cell = phase(sample.identity().tick()) * 2
            + usize::from(sample.side() == crate::ImitationSide::Dire);
        let (mut key, target) = Self::new(sample.teacher_action(), cell);
        match target {
            crate::ActionTarget::None => {}
            crate::ActionTarget::Entity(index) => {
                let row = sample.frame().units()[index.0];
                let relation = (0..4)
                    .find(|offset| row[crate::unit_feature::RELATION_START + offset] == 1.0)
                    .expect("entity relation");
                key.target = 1 + (row[crate::unit_feature::KIND_TOKEN] as usize - 1) * 4 + relation;
            }
            crate::ActionTarget::Point(index) => {
                key.point =
                    sample.frame().points()[index.0][crate::point_feature::SOURCE_TOKEN] as usize;
            }
        }
        assert!(key.target <= 48);
        assert!(key.point <= 8);
        key
    }
}

struct BranchSelector {
    capacity: usize,
    rng: PpoRng,
    keys: Vec<BranchKey>,
    counts: BTreeMap<BranchKey, (u64, Vec<usize>)>,
}

impl BranchSelector {
    fn new(capacity: usize, seed: u64) -> Self {
        assert!(capacity > 0);
        assert!(capacity <= MAX_IMITATION_SAMPLES);
        Self {
            capacity,
            rng: PpoRng::new(seed),
            keys: Vec::with_capacity(capacity),
            counts: BTreeMap::new(),
        }
    }

    fn destination(&mut self, key: BranchKey) -> Result<Option<usize>> {
        if !self.counts.contains_key(&key) && self.counts.len() == 8192 {
            return Err("conditional reservoir exceeds 8192 observed strata".into());
        }
        let entry = self.counts.entry(key).or_default();
        assert!(entry.0 < 108900 * 2 * 22);
        entry.0 += 1;
        let (seen, kept) = (entry.0, entry.1.len());
        let index = if self.keys.len() < self.capacity {
            self.keys.len()
        } else {
            let (&largest, (_, positions)) = self
                .counts
                .iter()
                .max_by_key(|(_, entry)| entry.1.len())
                .expect("nonempty strata");
            if positions.len() > kept + 1 {
                self.counts
                    .get_mut(&largest)
                    .expect("stratum")
                    .1
                    .pop()
                    .expect("retained row")
            } else {
                let draw = (self.rng.next_u64()? % seen) as usize;
                if draw >= kept {
                    return Ok(None);
                }
                return Ok(Some(self.counts[&key].1[draw]));
            }
        };
        if index == self.keys.len() {
            self.keys.push(key);
        } else {
            self.keys[index] = key;
        }
        self.counts
            .get_mut(&key)
            .expect("incoming stratum")
            .1
            .push(index);
        assert!(self.keys.len() <= self.capacity);
        Ok(Some(index))
    }
}

struct Reservoir {
    capacity: usize,
    seed: u64,
    selector: BranchSelector,
    samples: Vec<ImitationSample>,
    counts: [[u64; ActionKind::COUNT]; 5],
    counts_by_side: [[[u64; ActionKind::COUNT]; 5]; 2],
}

impl Reservoir {
    fn new(capacity: usize, seed: u64) -> Self {
        assert!((64..=MAX_IMITATION_SAMPLES).contains(&capacity));
        assert!(capacity.is_multiple_of(64));
        Self {
            capacity,
            seed,
            selector: BranchSelector::new(capacity, seed),
            samples: Vec::with_capacity(capacity),
            counts: [[0; ActionKind::COUNT]; 5],
            counts_by_side: [[[0; ActionKind::COUNT]; 5]; 2],
        }
    }
    fn dagger(capacity: usize, seed: u64) -> Self {
        assert!(matches!(capacity, 2048 | 4096));
        Self::new(capacity, seed)
    }
    fn consider(
        &mut self,
        seat: &mut NeuralSeat,
        space: &ActionSpace,
        action: StructuredAction,
        seed: u64,
        namespace: SeedNamespace,
        learner: Option<StructuredAction>,
    ) -> Result<()> {
        let kind = action.kind().index();
        let cell = phase(space.tick()) * 2 + usize::from(seat.tracker.team() == Team::Dire);
        self.counts[phase(space.tick())][kind] += 1;
        self.counts_by_side[cell % 2][phase(space.tick())][kind] += 1;
        let key = BranchKey::from_space(action, space, cell);
        if let Some(index) = self.selector.destination(key)? {
            let frame = seat.frame(space)?;
            let identity = SampleIdentity::from_frame(namespace, seed, 0, space.tick(), &frame)?;
            let sample = match learner {
                Some(learner) => ImitationSample::dagger(frame, space, learner, action, identity)?,
                None => ImitationSample::teacher(frame, space, action, identity)?,
            };
            assert_eq!(BranchKey::from_sample(&sample), key);
            if index == self.samples.len() {
                self.samples.push(sample);
            } else {
                self.samples[index] = sample;
            }
        }
        assert!(self.samples.len() <= self.capacity);
        Ok(())
    }
    fn into_samples(mut self) -> Result<Vec<ImitationSample>> {
        let (continued, mut samples): (Vec<_>, Vec<_>) = self
            .samples
            .drain(..)
            .partition(|sample| sample.teacher_action().kind() == ActionKind::Continue);
        let limit = continued.len().min(samples.len() / 2);
        let mut retained_continue = Vec::with_capacity(limit);
        let mut selector = CoverageSelector::new(limit, self.seed ^ 0xc017, false);
        for sample in continued {
            let cell = phase(sample.identity().tick()) * 2
                + usize::from(sample.side() == crate::ImitationSide::Dire);
            if let Some(index) = selector.destination(cell)? {
                if index == retained_continue.len() {
                    retained_continue.push(sample);
                } else {
                    retained_continue[index] = sample;
                }
            }
        }
        samples.extend(retained_continue);
        samples.sort_unstable_by_key(ImitationSample::identity);
        assert!(samples.len() <= self.capacity);
        Ok(samples)
    }
}

struct Collection<'a> {
    reservoir: &'a mut Reservoir,
    namespace: SeedNamespace,
    labeler: Option<Teacher>,
}

struct NeuralDataset {
    pool: ImitationPool,
    coverage: [TeacherCoverage; 2],
    games: Vec<NeuralGameReport>,
    seen: [[[u64; ActionKind::COUNT]; 5]; 3],
    seen_by_side: [[[[u64; ActionKind::COUNT]; 5]; 2]; 3],
    seen_branches: [BTreeMap<BranchKey, u64>; 3],
}

fn collect_dataset(
    config: &NeuralTrainingConfig,
    deadline: Option<Instant>,
) -> Result<NeuralDataset> {
    validate_config(config)?;
    let seeds = namespaces(config)?;
    let mut pool = ImitationPool::new(
        MAX_IMITATION_SAMPLES,
        config.seed | 1,
        seeds.clone(),
        TrainingScope::new(MapId(0), IMITATION_RULES_AUDIT_VERSION)?,
    )?;
    let mut coverage = [TeacherCoverage::new(), TeacherCoverage::new()];
    let mut games = Vec::with_capacity(config.training_games + 2);
    let mut seen = [[[0; ActionKind::COUNT]; 5]; 3];
    let mut seen_by_side = [[[[0; ActionKind::COUNT]; 5]; 2]; 3];
    let mut seen_branches = std::array::from_fn(|_| BTreeMap::new());
    for (split, namespace) in [
        SeedNamespace::Training,
        SeedNamespace::Validation,
        SeedNamespace::Promotion,
    ]
    .into_iter()
    .enumerate()
    {
        let mut reservoir = Reservoir::new(CAPACITIES[split], config.seed ^ (split as u64 + 71));
        let seeds = match split {
            0 => &seeds.training()[..config.training_games],
            1 => seeds.validation(),
            _ => seeds.promotion(),
        };
        for &seed in seeds {
            let game = run_game(
                None,
                seed,
                0,
                config.tick_limit,
                deadline,
                Some((&mut reservoir, namespace)),
            )?;
            println!(
                "expert seed={seed} ticks={} winner={:?} decisions={:?}",
                game.ticks, game.winner, game.decisions
            );
            games.push(game);
        }
        seen[split] = reservoir.counts;
        seen_by_side[split] = reservoir.counts_by_side;
        seen_branches[split] = reservoir
            .selector
            .counts
            .iter()
            .map(|(key, (seen, _))| (*key, *seen))
            .collect();
        for sample in reservoir.into_samples()? {
            if split > 0 {
                coverage[split - 1].record_represented_for(&sample)?;
            }
            assert!(pool.push(sample)?.is_none());
        }
    }
    Ok(NeuralDataset {
        pool,
        coverage,
        games,
        seen,
        seen_by_side,
        seen_branches,
    })
}

fn run_game(
    model: Option<&PolicyModel>,
    seed: u64,
    candidate: usize,
    limit: u32,
    deadline: Option<Instant>,
    collection: Option<(&mut Reservoir, SeedNamespace)>,
) -> Result<NeuralGameReport> {
    assert!(candidate < 2 && (2..=108900).contains(&limit));
    if model.is_some()
        && collection
            .as_ref()
            .is_some_and(|entry| entry.1 != SeedNamespace::Training)
    {
        return Err("DAgger may collect only Training namespace states".into());
    }
    let mut collection = collection.map(|(reservoir, namespace)| Collection {
        reservoir,
        namespace,
        labeler: model.is_some().then(Teacher::new),
    });
    let (mut arena, start) = Arena::new(ArenaConfig {
        map: MapId(0),
        seats: 2,
        seed,
    })?;
    let mut seats = [
        NeuralSeat::new(0, &start.messages[0])?,
        NeuralSeat::new(1, &start.messages[1])?,
    ];
    prepare_game_seats(&mut seats, model.map(|_| candidate), collection.is_some())?;
    let mut experts = std::array::from_fn::<_, 2, _>(|index| {
        (model.is_none() || index != candidate).then(Teacher::new)
    });
    let mut winner = None;
    for _ in 1..limit {
        if (arena.tick() == 1 || arena.tick().is_multiple_of(64))
            && deadline.is_some_and(|end| Instant::now() >= end)
        {
            return Err("neural match deadline exhausted".into());
        }
        let terminal = game_tick(
            &mut arena,
            &mut seats,
            &mut experts,
            model,
            seed,
            &mut collection,
        )?;
        if arena.tick() < limit {
            winner = terminal;
        }
        if terminal.is_some() {
            break;
        }
    }
    game_report(&seats, seed, candidate, arena.tick(), winner)
}

fn prepare_game_seats(
    seats: &mut [NeuralSeat; 2],
    candidate: Option<usize>,
    collecting: bool,
) -> Result<()> {
    assert!(candidate.is_none_or(|side| side < seats.len()));
    assert!(seats.iter().all(|seat| seat.sequence == 0));
    if let Some(side) = candidate {
        seats[side]
            .order_bookkeeping
            .enable_candidate(&seats[side].persistence)?;
    } else if collecting {
        for seat in seats {
            seat.order_bookkeeping.enable_observer(&seat.persistence)?;
        }
    }
    Ok(())
}

fn game_report(
    seats: &[NeuralSeat; 2],
    seed: u64,
    candidate: usize,
    ticks: u32,
    winner: Option<Team>,
) -> Result<NeuralGameReport> {
    assert!(candidate < 2);
    assert!((2..=108900).contains(&ticks));
    let summary = seats[candidate]
        .tracker
        .latest_summary()
        .ok_or("final summary missing")?;
    let progress = summary.destroyed_structures_present.then(|| {
        (
            i64::from(summary.enemy_structures_destroyed.min(64))
                - i64::from(summary.allied_structures_destroyed.min(64)),
            (summary.allied_structure_hp - summary.enemy_structure_hp).clamp(-100000, 100000),
        )
    });
    Ok(NeuralGameReport {
        seed,
        candidate,
        ticks,
        winner,
        progress,
        decisions: seats.each_ref().map(|seat| seat.decisions),
        orders: seats.each_ref().map(|seat| seat.sequence),
        order_fingerprints: seats.each_ref().map(|seat| seat.orders_hash.finish()),
        rejections: seats.each_ref().map(|seat| seat.rejections),
        kinds: seats.each_ref().map(|seat| seat.kinds),
        opening_kinds: seats.each_ref().map(|seat| seat.opening_kinds),
        left_fountain_area: seats.each_ref().map(|seat| seat.left_fountain_area),
    })
}

fn game_tick(
    arena: &mut Arena,
    seats: &mut [NeuralSeat; 2],
    experts: &mut [Option<Teacher>; 2],
    model: Option<&PolicyModel>,
    seed: u64,
    collection: &mut Option<Collection<'_>>,
) -> Result<Option<Team>> {
    assert!(arena.tick() > 0);
    assert!(arena.tick() < 108900);
    let mut requests = [None, None];
    if (arena.tick() - 1).is_multiple_of(3) {
        for (index, seat) in seats.iter_mut().enumerate() {
            let (action, space) = if let Some(expert) = &mut experts[index] {
                expert.decide(&seat.tracker, &seat.persistence, &seat.readiness)?
            } else {
                seat.neural_choice(model.ok_or("learner model missing")?)?
            };
            if let Some(collection) = collection {
                collect_decision(
                    collection,
                    seat,
                    &space,
                    action,
                    seed,
                    experts[index].is_none(),
                )?;
            }
            if let Some(issued) = seat.issue(action, &space)? {
                if let Some(expert) = &mut experts[index] {
                    expert.note_sent(seat.sequence, issued, space.tick());
                } else if let Some(collection) = collection {
                    note_labeler_send(collection, seat.sequence, issued, space.tick());
                }
                requests[index] = Some(Request {
                    seq: seat.sequence,
                    unit: issued.unit,
                    order: issued.order,
                });
            }
        }
    }
    let step = arena.step(&requests)?;
    observe_game_tick(seats, experts, &step.messages, seed, collection)
}

fn observe_game_tick(
    seats: &mut [NeuralSeat; 2],
    experts: &mut [Option<Teacher>; 2],
    messages: &[Vec<ServerMsg>],
    seed: u64,
    collection: &mut Option<Collection<'_>>,
) -> Result<Option<Team>> {
    assert_eq!(messages.len(), 2);
    let mut terminal = [None, None];
    for index in 0..2 {
        for message in &messages[index] {
            if let ServerMsg::OrderRejected { seq, .. } = message
                && let Some(expert) = &mut experts[index]
            {
                expert.note_rejected(*seq);
                return Err(format!(
                    "expert rejected order seed={seed} seat={index} sequence={seq}"
                )
                .into());
            }
            if experts[index].is_none()
                && let ServerMsg::OrderRejected { seq, .. } = message
                && let Some(collection) = collection
                && let Some(labeler) = &mut collection.labeler
            {
                labeler.note_rejected(*seq);
            }
        }
        terminal[index] = seats[index].observe(&messages[index]).map_err(|error| {
            format!(
                "neural arena seed={seed} seat={index} controller={}: {error}",
                if experts[index].is_some() {
                    "reference_teacher"
                } else {
                    "neural"
                }
            )
        })?;
    }
    assert_eq!(terminal[0], terminal[1]);
    Ok(terminal[0])
}

fn collect_decision(
    collection: &mut Collection<'_>,
    seat: &mut NeuralSeat,
    space: &ActionSpace,
    learner: StructuredAction,
    seed: u64,
    neural: bool,
) -> Result<()> {
    if let Some(labeler) = &mut collection.labeler {
        if !neural {
            return Ok(());
        }
        assert_eq!(collection.namespace, SeedNamespace::Training);
        let expert = labeler
            .decide(&seat.tracker, &seat.persistence, &seat.readiness)?
            .0;
        assert!(space.allows(expert));
        collection.reservoir.consider(
            seat,
            space,
            expert,
            seed,
            collection.namespace,
            Some(learner),
        )
    } else {
        assert!(!neural);
        collection
            .reservoir
            .consider(seat, space, learner, seed, collection.namespace, None)
    }
}

fn note_labeler_send(
    collection: &mut Collection<'_>,
    sequence: u32,
    issued: crate::IssuedOrder,
    tick: u32,
) {
    assert!(sequence > 0);
    assert!(tick > 0);
    if let Some(labeler) = &mut collection.labeler {
        labeler.note_sent(sequence, issued, tick);
    }
}

fn evaluate_pair(
    model: &PolicyModel,
    seed: u64,
    limit: u32,
    deadline: Option<Instant>,
) -> Result<Vec<NeuralGameReport>> {
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(2);
        for side in 0..2 {
            workers.push(
                std::thread::Builder::new()
                    .name(format!("neural-eval-{side}"))
                    .stack_size(8 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        run_game(Some(model), seed, side, limit, deadline, None)
                    })?,
            );
        }
        let completed = workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .unwrap_or_else(|_| Err("neural evaluation worker panicked".into()))
            })
            .collect::<Vec<_>>();
        assert_eq!(completed.len(), 2);
        completed.into_iter().collect()
    })
}

fn gameplay_rank(games: &[NeuralGameReport]) -> (i64, i64, i64, i64) {
    assert_eq!(games.len(), 2);
    let mut rank = (0, 0, 0, 0);
    for game in games {
        rank.0 += i64::from(game.winner == Some([Team::Radiant, Team::Dire][game.candidate]));
        rank.1 -= i64::from(game.rejections[game.candidate]);
        if let Some((structures, hp)) = game.progress {
            rank.2 += structures;
            rank.3 += hp;
        }
    }
    rank
}

fn namespaces(config: &NeuralTrainingConfig) -> Result<SeedNamespaces> {
    Ok(SeedNamespaces::new(
        (0..config.training_games)
            .map(|index| config.seed + index as u64)
            .chain((0..config.dagger_rounds).map(|round| config.seed + 50 + round as u64))
            .collect(),
        vec![config.seed + 100],
        vec![config.seed + 200],
    )?)
}

fn validate_config(config: &NeuralTrainingConfig) -> Result<()> {
    if config.initial_weights.is_some() && config.initialize_selected_m10.is_some() {
        return Err("choose only one of initial_weights and initialize_selected_m10".into());
    }
    if !(1..=20).contains(&config.training_games)
        || !(1..=64).contains(&config.epochs)
        || !(2..=108900).contains(&config.tick_limit)
        || config.seed.checked_add(201).is_none()
        || config.wall_time.is_zero()
        || config.wall_time > Duration::from_secs(2700)
        || config.dagger_rounds > 2
        || !(1..=64).contains(&config.dagger_epochs)
    {
        return Err("Map0 training requires 1..20 expert games, 1..64 epochs per stage, 0..2 DAgger rounds, 2..108900 ticks, <=2700 seconds and nonoverflowing disjoint seeds".into());
    }
    Ok(())
}

fn initialize_model(config: &NeuralTrainingConfig) -> Result<(PolicyModel, Initialization)> {
    validate_config(config)?;
    if let Some(source) = &config.initialize_selected_m10 {
        let (model, sha256) = TrainingArtifact::initialize_selected_m10_for_training(
            source,
            config.seed,
            config.device,
        )?;
        let sha256 = sha256.iter().map(|byte| format!("{byte:02x}")).collect();
        return Ok((
            model,
            Initialization::SelectedM10 {
                source: source.clone(),
                sha256,
            },
        ));
    }
    let model = PolicyModel::fresh_on(config.seed, config.device)?;
    let initialization = if let Some(source) = &config.initial_weights {
        TrainingArtifact::load_runtime_weights(&model, source)?;
        let mut bytes = Vec::new();
        File::open(source.join("drysua.weights.safetensors"))?
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 8 * 1024 * 1024 {
            return Err("current initialization source exceeds 8 MiB provenance read bound".into());
        }
        let sha256 = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Initialization::CurrentWeights {
            source: source.clone(),
            sha256,
        }
    } else {
        Initialization::Fresh
    };
    Ok((model, initialization))
}

fn phase(tick: u32) -> usize {
    if tick < 900 {
        0
    } else {
        (1 + (tick - 900) / 18000).min(4) as usize
    }
}

fn write_dataset(data: &NeuralDataset, output: &Path) -> Result<()> {
    write_branch_distribution(data, output)?;
    let mut retained = [[[0u64; ActionKind::COUNT]; 5]; 3];
    let mut retained_by_side = [[[[0u64; ActionKind::COUNT]; 5]; 2]; 3];
    let mut targets = [[0u64; 13]; 3];
    let mut hash = Sha256::new();
    let mut index = File::create(output.join("dataset-index.txt"))?;
    for row in 0..data.pool.len() {
        let sample = data.pool.get(row).ok_or("pool sample missing")?;
        let split = match sample.identity().namespace() {
            SeedNamespace::Training => 0,
            SeedNamespace::Validation => 1,
            SeedNamespace::Promotion => 2,
        };
        retained[split][phase(sample.identity().tick())][sample.teacher_action().kind().index()] +=
            1;
        retained_by_side[split][usize::from(sample.side() == crate::ImitationSide::Dire)]
            [phase(sample.identity().tick())][sample.teacher_action().kind().index()] += 1;
        if sample.target().entity_pointer.active {
            let kind = sample.frame().units()[sample.target().entity_pointer.selected]
                [crate::unit_feature::KIND_TOKEN] as usize;
            assert!(kind < 13);
            targets[split][kind] += 1;
        }
        let identity = hash_sample(sample, &mut hash);
        write!(index, "{identity}")?;
    }
    let identity = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    fs::write(
        output.join("dataset.txt"),
        format!(
            "sampling=conditional_branch_stratified_NOT_population_accuracy\ncontinue_max_fraction=1/3\ncapacities={CAPACITIES:?} dagger_capacity={DAGGER_CAPACITY}\npool_binding={:?}\nframe_inline_bytes={} sample_inline_bytes={} samples={}\nseen={:?}\nretained={retained:?}\nentity_target_kinds={targets:?}\ndataset_sha256={identity}\nexpert_games={:?}\nseen_by_side={:?}\nretained_by_side={retained_by_side:?}\n",
            data.pool.binding(),
            std::mem::size_of::<FeatureFrame>(),
            std::mem::size_of::<ImitationSample>(),
            data.pool.len(),
            data.seen,
            data.games,
            data.seen_by_side
        ),
    )?;
    Ok(())
}

fn write_branch_distribution(data: &NeuralDataset, output: &Path) -> Result<()> {
    let mut retained: [BTreeMap<BranchKey, usize>; 3] = std::array::from_fn(|_| BTreeMap::new());
    assert!(data.pool.len() <= MAX_IMITATION_SAMPLES);
    for row in 0..data.pool.len() {
        let sample = data.pool.get(row).ok_or("branch sample missing")?;
        let split = match sample.identity().namespace() {
            SeedNamespace::Training => 0,
            SeedNamespace::Validation => 1,
            SeedNamespace::Promotion => 2,
        };
        *retained[split]
            .entry(BranchKey::from_sample(sample))
            .or_default() += 1;
    }
    let mut file = File::create(output.join("branch-distribution.txt"))?;
    writeln!(
        file,
        "body=0:Hero,1:Courier,2:none target=1+UnitKind*4+relation(Own,Allied,Enemy,Neutral) point=0:none,1:tactical,2:static_tree,3:planted_tree,4:building,5:fountain,6:tower,7:predicted_hero,8:predicted_creep cell=phase*2+side slot=0:none,otherwise_slot+1(or_shop+1,swap_pair+1)"
    )?;
    for (split, counts) in data.seen_branches.iter().enumerate() {
        assert!(counts.len() <= 8192);
        for (key, seen) in counts {
            writeln!(
                file,
                "split={split} {key:?} seen={seen} retained={}",
                retained[split].get(key).copied().unwrap_or(0)
            )?;
        }
        let courier_seen: u64 = counts
            .iter()
            .filter(|(key, _)| key.body == 1)
            .map(|(_, count)| count)
            .sum();
        let courier_retained: usize = retained[split]
            .iter()
            .filter(|(key, _)| key.body == 1)
            .map(|(_, count)| count)
            .sum();
        writeln!(
            file,
            "split={split} courier_seen={courier_seen} courier_retained={courier_retained}"
        )?;
    }
    Ok(())
}

fn write_branch_accuracy(
    model: &PolicyModel,
    data: &NeuralDataset,
    namespace: SeedNamespace,
    output: &Path,
) -> Result<()> {
    let samples = (0..data.pool.len())
        .filter_map(|index| data.pool.get(index))
        .filter(|sample| sample.identity().namespace() == namespace)
        .collect::<Vec<_>>();
    assert!(!samples.is_empty());
    assert!(samples.len() <= MAX_IMITATION_SAMPLES);
    let mut branches: BTreeMap<BranchKey, [crate::AgreementCount; 13]> = BTreeMap::new();
    for batch in samples.chunks(64) {
        for (sample, prediction) in batch.iter().zip(model.behavioral_predictions(batch)?) {
            let counts = branches.entry(BranchKey::from_sample(sample)).or_default();
            let target = sample.target();
            let heads = [
                head_match(&target.kind, Some(prediction.kind)),
                head_match(&target.controlled, prediction.controlled),
                head_match(&target.ability, prediction.ability),
                head_match(&target.item, prediction.item),
                head_match(&target.swap, prediction.swap),
                head_match(&target.learn, prediction.learn),
                head_match(&target.shop, prediction.shop),
                head_match(&target.loot, prediction.loot),
                head_match(&target.target_mode, prediction.target_mode),
                head_match(&target.put_mode, prediction.put_mode),
                head_match(&target.entity_pointer, prediction.entity_pointer),
                head_match(&target.point_pointer, prediction.point_pointer),
            ];
            let mut full = true;
            for (index, matching) in heads.into_iter().enumerate() {
                if let Some(matching) = matching {
                    counts[index].total += 1;
                    counts[index].matching += usize::from(matching);
                    full &= matching;
                }
            }
            counts[12].total += 1;
            counts[12].matching += usize::from(full);
        }
    }
    let mut file = File::create(output)?;
    writeln!(
        file,
        "teacher_forced_conditional_accuracy_not_free_running=true heads=kind,body,ability,item,swap,learn,shop,loot,target_mode,put_mode,entity,point,full"
    )?;
    for (key, counts) in branches {
        writeln!(file, "{key:?} {counts:?}")?;
    }
    Ok(())
}

fn head_match<const N: usize>(
    target: &crate::HeadTarget<N>,
    prediction: Option<usize>,
) -> Option<bool> {
    assert!(!target.active || target.selected < N);
    assert!(prediction.is_none_or(|index| index < N));
    target.active.then_some(prediction == Some(target.selected))
}

fn hash_sample(sample: &ImitationSample, hash: &mut Sha256) -> String {
    let frame = sample.frame();
    assert!(frame.is_finite());
    for values in [
        frame.global().as_slice(),
        frame.history().as_flattened(),
        frame.policy_history().as_flattened(),
        frame.units().as_flattened(),
        frame.own_units().as_flattened(),
        frame.remembered_units().as_flattened(),
        frame.points().as_flattened(),
        frame.abilities().as_flattened(),
        frame.items().as_flattened(),
        frame.projectiles().as_flattened(),
        frame.loot().as_flattened(),
        frame.map().as_slice(),
    ] {
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    let identity = format!(
        "{:?} source={:?} learner={:?} {:?} {:?}\n",
        sample.identity(),
        sample.source(),
        sample.learner_action(),
        sample.teacher_action(),
        sample.target()
    );
    hash.update(identity.as_bytes());
    identity
}

fn real_continue(data: &NeuralDataset, split: usize) -> f64 {
    assert!(split < 3);
    let total: u64 = data.seen[split].iter().flatten().sum();
    assert!(total > 0);
    data.seen[split].iter().map(|row| row[0]).sum::<u64>() as f64 / total as f64
}

#[cfg(test)]
#[path = "tests/neural_training.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/neural_training_contract.rs"]
mod training_contract;
