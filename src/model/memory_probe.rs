//! Offline frozen-encoder temporal-utility screen, not a recurrent PPO policy.
//! Coordinator entry: `model.memory_probe(&batch)` on a CPU policy and 40 complete
//! single-episode streams, prepared with gamma=lambda=1 and unmodified game rewards.
//! The final <=128 consecutive retained rows of each episode form one sequence.
//! This is the already-sampled stream (normally stride eight, about 0.8s), not 100ms
//! recurrent deployment. Hidden state starts at zero at each selected window.
//! Every fifth adjacent pair of sorted streams is held out: 32 train / 8 held out.
//! Adjacent-pair seed provenance and unmodified reward provenance belong to the caller.
//! Ending-window selection can reward a learned position counter, not only useful history.
//! Rotated-state evaluation mixes past states across held-out lanes, never future inputs.
//! Output is JSON lines on stdout; no checkpoints, files, or production state change.
//! Coordinator test filter: `model::memory_probe::`; this module has no rollout runner.

use std::collections::BTreeMap;
use std::time::Instant;

use sha2::{Digest, Sha256};

use super::*;

const EPISODES: usize = 40;
const WINDOW: usize = 128;
const HIDDEN: usize = 32;
const CONTROL: usize = 108;
const SEQUENCES: usize = 4;
const UNROLL: usize = 16;
const SEED: u64 = 0x4d45_4d34_3238;
const LEARNING_RATE: f64 = 0.02;
const _: () = assert!(EPISODES * WINDOW * TRUNK_WIDTH * 4 < 16 * 1024 * 1024);
const _: () = assert!(32 / SEQUENCES * (WINDOW / UNROLL) == 64);

impl PolicyModel {
    /// Prints a bounded offline GRU/control screen; requires CPU, 40 full gamma-one MC episodes.
    /// Source weights stay read-only; auxiliary weights are discarded, never model24 artifacts.
    /// Call from the coordinator's cfg(test) lab after collection, before changing this policy.
    pub(crate) fn memory_probe(&self, batch: &crate::PpoBatch) -> Result<(), ModelError> {
        if !self.tensor_device().is_cpu() {
            return Err(ModelError::InvalidModelState(
                "memory probe requires a CPU actor snapshot",
            ));
        }
        let started = Instant::now();
        let _guard = self.read_parameter_lock()?;
        if self.policy_identity_locked() != batch.policy() {
            return Err(ModelError::InvalidModelState(
                "memory probe source policy differs from batch",
            ));
        }
        let before = source_hash(self)?;
        let result = (|| {
            let mut episodes = collect_episodes(batch)?;
            encode_episodes(self, batch, &mut episodes)?;
            let normalization = Normalization::fit(&episodes)?;
            normalization.apply(&mut episodes);
            let rows = episodes
                .iter()
                .map(|episode| episode.inputs.len())
                .sum::<usize>();
            let ticks_min = episodes
                .iter()
                .map(|episode| episode.ticks_min)
                .min()
                .unwrap();
            let ticks_max = episodes
                .iter()
                .map(|episode| episode.ticks_max)
                .max()
                .unwrap();
            println!(
                concat!(
                    "{{\"memory_probe\":\"config\",\"seed\":{},\"objective\":\"full_gamma1_mc_return\",",
                    "\"window\":\"last_128_retained_rows_zero_initial_state\",\"train_episodes\":32,\"heldout_episodes\":8,",
                    "\"rows\":{},\"embedding_bytes\":{},\"scanned_rows\":{},\"ticks_min\":{},\"ticks_max\":{},",
                    "\"tbptt\":16,\"sequence_batch\":4,\"max_steps_per_model\":64,\"sgd_lr\":0.02,\"gradient_clip\":1,",
                    "\"input_normalization\":\"train_only_sd_floor_0.001_clip10\",\"target_scale\":{},",
                    "\"source_parameters_sha256\":\"{}\",\"extraction_ms\":{},\"deployed_recurrence\":false}}"
                ),
                SEED,
                rows,
                rows * TRUNK_WIDTH * 4,
                batch.len(),
                ticks_min,
                ticks_max,
                normalization.target_scale,
                before,
                started.elapsed().as_millis()
            );
            run_screen(&episodes, &normalization)
        })();
        let after = source_hash(self)?;
        if before != after {
            return Err(ModelError::InvalidModelState(
                "memory probe mutated source weights",
            ));
        }
        println!(
            "{{\"memory_probe\":\"source\",\"unchanged\":true,\"parameters_sha256\":\"{after}\",\"elapsed_ms\":{},\"success\":{}}}",
            started.elapsed().as_millis(),
            result.is_ok()
        );
        result
    }
}

struct Episode {
    stream: usize,
    train: bool,
    indices: Vec<usize>,
    inputs: Vec<[f32; TRUNK_WIDTH]>,
    returns: Vec<f32>,
    total_return: f64,
    expected_decision: u32,
    rows: usize,
    ticks_min: u32,
    ticks_max: u32,
}

impl Episode {
    fn empty(stream: usize) -> Self {
        Self {
            stream,
            train: false,
            indices: Vec::new(),
            inputs: Vec::new(),
            returns: Vec::new(),
            total_return: 0.0,
            expected_decision: 0,
            rows: 0,
            ticks_min: u32::MAX,
            ticks_max: 0,
        }
    }

    fn observe_reverse(
        &mut self,
        index: usize,
        sample: PpoPreparedSample,
    ) -> Result<(), ModelError> {
        let transition = sample.transition;
        if transition.terminal != (self.rows == 0)
            || (self.rows > 0 && transition.decision.checked_add(1) != Some(self.expected_decision))
            || transition.ticks == 0
            || !transition.reward.is_finite()
            || !sample.return_value.is_finite()
        {
            return Err(ModelError::InvalidModelState(
                "memory probe requires contiguous complete single-episode streams",
            ));
        }
        self.total_return += f64::from(transition.reward);
        if (self.total_return - f64::from(sample.return_value)).abs()
            > 1.0e-5 * (1.0 + self.total_return.abs())
        {
            return Err(ModelError::InvalidModelState(
                "memory probe targets are not full gamma-one Monte Carlo returns",
            ));
        }
        if self.indices.len() < WINDOW {
            self.indices.push(index);
            self.returns.push(sample.return_value);
            self.ticks_min = self.ticks_min.min(transition.ticks);
            self.ticks_max = self.ticks_max.max(transition.ticks);
        }
        self.expected_decision = transition.decision;
        self.rows += 1;
        Ok(())
    }
}

fn collect_episodes(batch: &crate::PpoBatch) -> Result<Vec<Episode>, ModelError> {
    if batch.is_empty() || batch.len() > EPISODES * crate::MAP2_RETAINED_DECISIONS {
        return Err(ModelError::InvalidModelState(
            "memory probe batch row bound",
        ));
    }
    let mut streams = BTreeMap::new();
    for index in (0..batch.len()).rev() {
        let sample = batch
            .sample(index)
            .map_err(|error| ModelError::Backend(error.to_string()))?;
        let stream = sample.transition.stream;
        if !streams.contains_key(&stream) && streams.len() == EPISODES {
            return Err(ModelError::InvalidModelState(
                "memory probe exceeds 40 streams",
            ));
        }
        streams
            .entry(stream)
            .or_insert_with(|| Episode::empty(stream))
            .observe_reverse(index, sample)?;
    }
    let mut episodes = streams.into_values().collect::<Vec<_>>();
    for episode in &mut episodes {
        if episode.expected_decision != 0 {
            return Err(ModelError::InvalidModelState(
                "memory probe episode does not start at decision zero",
            ));
        }
        episode.indices.reverse();
        episode.returns.reverse();
    }
    assign_split(&mut episodes)?;
    Ok(episodes)
}

fn assign_split(episodes: &mut [Episode]) -> Result<(), ModelError> {
    if episodes.len() != EPISODES {
        return Err(ModelError::InvalidModelState(
            "memory probe requires 40 complete episode streams",
        ));
    }
    episodes.sort_by_key(|episode| episode.stream);
    for (index, episode) in episodes.iter_mut().enumerate() {
        episode.train = (index / 2) % 5 != 4;
    }
    Ok(())
}

fn encode_episodes(
    model: &PolicyModel,
    batch: &crate::PpoBatch,
    episodes: &mut [Episode],
) -> Result<(), ModelError> {
    assert_eq!(episodes.len(), EPISODES);
    for episode in episodes {
        assert!(episode.indices.len() <= WINDOW);
        for indices in episode.indices.chunks(MODEL_TRAINING_BATCH) {
            let mut frames = Vec::with_capacity(indices.len());
            for index in indices {
                // Materialize only a single encoder microbatch, never all dense frames.
                let sample = batch
                    .sample(*index)
                    .map_err(|error| ModelError::Backend(error.to_string()))?;
                frames.push(sample.transition.frame);
            }
            validate_batch(&frames)?;
            let state = model.forward_frames(&frames)?;
            validate_tensor_finite("memory probe embedding", &state.trunk)?;
            for row in state.trunk.detach().to_vec2::<f32>()? {
                episode.inputs.push(
                    row.try_into().map_err(|_| {
                        ModelError::InvalidModelState("memory probe embedding width")
                    })?,
                );
            }
        }
        assert_eq!(episode.inputs.len(), episode.returns.len());
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
struct Normalization {
    mean: [f64; TRUNK_WIDTH],
    scale: [f64; TRUNK_WIDTH],
    target_mean: f64,
    target_scale: f64,
}

impl Normalization {
    fn fit(episodes: &[Episode]) -> Result<Self, ModelError> {
        let mut result = Self {
            mean: [0.0; TRUNK_WIDTH],
            scale: [0.0; TRUNK_WIDTH],
            target_mean: 0.0,
            target_scale: 0.0,
        };
        let mut count = 0;
        for episode in episodes.iter().filter(|episode| episode.train) {
            assert_eq!(episode.inputs.len(), episode.returns.len());
            assert!(episode.inputs.len() <= WINDOW);
            for (input, target) in episode.inputs.iter().zip(&episode.returns) {
                count += 1;
                for (index, value) in input.iter().enumerate() {
                    result.mean[index] += f64::from(*value);
                    result.scale[index] += f64::from(*value).powi(2);
                }
                result.target_mean += f64::from(*target);
                result.target_scale += f64::from(*target).powi(2);
            }
        }
        if count == 0 {
            return Err(ModelError::InvalidModelState(
                "memory probe has no training rows",
            ));
        }
        for index in 0..TRUNK_WIDTH {
            result.mean[index] /= count as f64;
            result.scale[index] = (result.scale[index] / count as f64 - result.mean[index].powi(2))
                .max(0.0)
                .sqrt()
                .max(0.001);
        }
        result.target_mean /= count as f64;
        result.target_scale = (result.target_scale / count as f64 - result.target_mean.powi(2))
            .max(0.0)
            .sqrt()
            .max(0.001);
        Ok(result)
    }

    fn apply(&self, episodes: &mut [Episode]) {
        for episode in episodes {
            for input in &mut episode.inputs {
                for (index, value) in input.iter_mut().enumerate() {
                    *value = ((f64::from(*value) - self.mean[index]) / self.scale[index])
                        .clamp(-10.0, 10.0) as f32;
                }
            }
        }
    }
}

struct ProbeNetwork {
    input: Linear,
    recurrent: Option<Linear>,
    output: Linear,
}

impl ProbeNetwork {
    fn new(recurrent: bool) -> Result<Self, ModelError> {
        let mut generator = Initializer::new(SEED);
        let width = if recurrent { 3 * HIDDEN } else { CONTROL };
        let input = Linear::fresh_with_gain(TRUNK_WIDTH, width, &mut generator, &Device::Cpu, 1.0)?;
        let recurrent = if recurrent {
            Some(Linear::fresh_with_gain(
                HIDDEN,
                width,
                &mut generator,
                &Device::Cpu,
                0.5,
            )?)
        } else {
            None
        };
        let output = Linear::fresh_with_gain(
            if recurrent.is_some() { HIDDEN } else { CONTROL },
            1,
            &mut generator,
            &Device::Cpu,
            0.1,
        )?;
        Ok(Self {
            input,
            recurrent,
            output,
        })
    }

    fn parameters(&self) -> Vec<&Var> {
        let mut variables = vec![
            &self.input.weight,
            &self.input.bias,
            &self.output.weight,
            &self.output.bias,
        ];
        if let Some(recurrent) = &self.recurrent {
            variables.extend([&recurrent.weight, &recurrent.bias]);
        }
        variables
    }

    fn step(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        reset: bool,
    ) -> Result<(Tensor, Tensor), ModelError> {
        assert_eq!(input.dim(1)?, TRUNK_WIDTH);
        assert_eq!(hidden.dims(), [input.dim(0)?, HIDDEN]);
        let input = self.input.forward(input)?;
        let Some(recurrent) = &self.recurrent else {
            return Ok((self.output.forward(&input.relu()?)?, hidden.clone()));
        };
        let hidden = if reset {
            hidden.zeros_like()?
        } else {
            hidden.clone()
        };
        let prior = recurrent.forward(&hidden)?;
        let reset_gate = (&input.narrow(1, 0, HIDDEN)? + &prior.narrow(1, 0, HIDDEN)?)?
            .affine(0.5, 0.0)?
            .tanh()?
            .affine(0.5, 0.5)?;
        let update_gate = (&input.narrow(1, HIDDEN, HIDDEN)?
            + &prior.narrow(1, HIDDEN, HIDDEN)?)?
            .affine(0.5, 0.0)?
            .tanh()?
            .affine(0.5, 0.5)?;
        let candidate = (&input.narrow(1, 2 * HIDDEN, HIDDEN)?
            + reset_gate.mul(&prior.narrow(1, 2 * HIDDEN, HIDDEN)?)?)?
        .tanh()?;
        let next = (update_gate.mul(&hidden)? + update_gate.affine(-1.0, 1.0)?.mul(&candidate)?)?;
        Ok((self.output.forward(&next)?, next))
    }
}

struct StepBatch {
    input: Tensor,
    target: Tensor,
    valid: Tensor,
    count: usize,
}

fn step_batch(
    episodes: &[&Episode],
    offset: usize,
    normalization: &Normalization,
) -> Result<StepBatch, ModelError> {
    assert!(!episodes.is_empty());
    assert!(episodes.len() <= SEQUENCES);
    let mut inputs = vec![0.0; episodes.len() * TRUNK_WIDTH];
    let mut targets = vec![0.0; episodes.len()];
    let mut valid = vec![0.0f32; episodes.len()];
    let mut count = 0;
    for (index, episode) in episodes.iter().enumerate() {
        if let Some(input) = episode.inputs.get(offset) {
            inputs[index * TRUNK_WIDTH..(index + 1) * TRUNK_WIDTH].copy_from_slice(input);
            targets[index] = ((f64::from(episode.returns[offset]) - normalization.target_mean)
                / normalization.target_scale) as f32;
            valid[index] = 1.0;
            count += 1;
        }
    }
    let shape = (episodes.len(), 1);
    Ok(StepBatch {
        input: Tensor::from_vec(inputs, (episodes.len(), TRUNK_WIDTH), &Device::Cpu)?,
        target: Tensor::from_vec(targets, shape, &Device::Cpu)?,
        valid: Tensor::from_vec(valid, shape, &Device::Cpu)?,
        count,
    })
}

fn preserve_padding(
    previous: &Tensor,
    next: &Tensor,
    valid: &Tensor,
) -> Result<Tensor, ModelError> {
    Ok((next.broadcast_mul(valid)? + previous.broadcast_mul(&valid.affine(-1.0, 1.0)?)?)?)
}

fn train(
    network: &ProbeNetwork,
    episodes: &[&Episode],
    normalization: &Normalization,
) -> Result<usize, ModelError> {
    assert_eq!(episodes.len(), 32);
    assert!(episodes.iter().all(|episode| episode.train));
    let mut steps = 0;
    for group in episodes.chunks(SEQUENCES) {
        let mut hidden = Tensor::zeros((group.len(), HIDDEN), DType::F32, &Device::Cpu)?;
        for offset in (0..WINDOW).step_by(UNROLL) {
            let mut loss = Tensor::zeros((), DType::F32, &Device::Cpu)?;
            let mut count = 0;
            for index in offset..offset + UNROLL {
                let batch = step_batch(group, index, normalization)?;
                let (prediction, next) = network.step(&batch.input, &hidden, index == 0)?;
                hidden = preserve_padding(&hidden, &next, &batch.valid)?;
                loss = (loss
                    + (prediction - batch.target)?
                        .sqr()?
                        .mul(&batch.valid)?
                        .sum_all()?)?;
                count += batch.count;
            }
            hidden = hidden.detach();
            if count > 0 {
                sgd(network, &loss.affine(1.0 / count as f64, 0.0)?)?;
                steps += 1;
            }
        }
    }
    assert!(steps <= 64);
    Ok(steps)
}

fn sgd(network: &ProbeNetwork, loss: &Tensor) -> Result<(), ModelError> {
    if !loss.to_scalar::<f32>()?.is_finite() {
        return Err(ModelError::NonFiniteLoss);
    }
    let gradients = loss.backward()?;
    let variables = network.parameters();
    let mut norm = 0.0f64;
    let mut updates = Vec::with_capacity(variables.len());
    for variable in &variables {
        let gradient = gradients
            .get(variable.as_tensor())
            .ok_or(ModelError::InvalidModelState(
                "memory probe missing auxiliary gradient",
            ))?;
        for value in gradient.flatten_all()?.to_vec1::<f32>()? {
            norm += f64::from(value).powi(2);
        }
        updates.push(gradient.clone());
    }
    if !norm.is_finite() {
        return Err(ModelError::NonFiniteOptimizerNorm);
    }
    let rate = LEARNING_RATE / norm.sqrt().max(1.0);
    for (variable, update) in variables.iter().zip(&mut updates) {
        *update = (variable.as_tensor() - update.affine(rate, 0.0)?)?.detach();
        validate_tensor_finite("memory probe auxiliary update", update)?;
    }
    for (variable, update) in variables.iter().zip(updates) {
        variable.set(&update)?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum History {
    Full,
    Zero,
    Shuffled,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    rows: usize,
    target: f64,
    target_square: f64,
    error: f64,
    error_square: f64,
}

impl Metrics {
    fn observe(&mut self, target: f64, prediction: f64) -> Result<(), ModelError> {
        if !prediction.is_finite() || !target.is_finite() {
            return Err(ModelError::NonFiniteLoss);
        }
        self.rows += 1;
        self.target += target;
        self.target_square += target * target;
        self.error += target - prediction;
        self.error_square += (target - prediction).powi(2);
        Ok(())
    }

    fn print(
        &self,
        model: &str,
        split: &str,
        history: &str,
        episodes: usize,
        stream: Option<usize>,
    ) {
        assert!(self.rows > 0);
        let rows = self.rows as f64;
        let variance = (self.target_square / rows - (self.target / rows).powi(2)).max(0.0);
        let error_variance = (self.error_square / rows - (self.error / rows).powi(2)).max(0.0);
        let explained = if variance > 1.0e-12 {
            (1.0 - error_variance / variance).to_string()
        } else {
            "null".to_owned()
        };
        let stream = stream.map_or_else(|| "null".to_owned(), |stream| stream.to_string());
        println!(
            concat!(
                "{{\"memory_probe\":\"metrics\",\"model\":\"{}\",\"split\":\"{}\",\"history\":\"{}\",",
                "\"episodes\":{},\"stream\":{},\"rows_not_independent\":{},\"return_mean\":{},\"return_variance\":{},",
                "\"rmse\":{},\"explained_variance\":{}}}"
            ),
            model,
            split,
            history,
            episodes,
            stream,
            self.rows,
            self.target / rows,
            variance,
            (self.error_square / rows).sqrt(),
            explained
        );
    }
}

fn evaluate(
    network: &ProbeNetwork,
    episodes: &[&Episode],
    normalization: &Normalization,
    history: History,
) -> Result<Vec<Metrics>, ModelError> {
    let mut metrics = Vec::with_capacity(episodes.len());
    for group in episodes.chunks(SEQUENCES) {
        let mut group_metrics = vec![Metrics::default(); group.len()];
        let mut hidden = Tensor::zeros((group.len(), HIDDEN), DType::F32, &Device::Cpu)?;
        let permutation = (0..group.len())
            .map(|index| ((index + 1) % group.len()) as u32)
            .collect::<Vec<_>>();
        let permutation = Tensor::from_vec(permutation, group.len(), &Device::Cpu)?;
        let length = group
            .iter()
            .map(|episode| episode.inputs.len())
            .max()
            .unwrap_or(0);
        for index in 0..length {
            if history == History::Shuffled {
                hidden = hidden.index_select(&permutation, 0)?;
            }
            let batch = step_batch(group, index, normalization)?;
            let (prediction, next) = network.step(
                &batch.input,
                &hidden,
                index == 0 || history == History::Zero,
            )?;
            hidden = preserve_padding(&hidden, &next, &batch.valid)?.detach();
            let predictions = prediction.flatten_all()?.to_vec1::<f32>()?;
            for (row, episode) in group.iter().enumerate() {
                if let Some(target) = episode.returns.get(index) {
                    group_metrics[row].observe(
                        f64::from(*target),
                        normalization.target_mean
                            + normalization.target_scale * f64::from(predictions[row]),
                    )?;
                }
            }
        }
        metrics.extend(group_metrics);
    }
    Ok(metrics)
}

fn report(model: &str, episodes: &[&Episode], metrics: &[Metrics], history: History) {
    assert_eq!(episodes.len(), metrics.len());
    assert!(!episodes.is_empty());
    let split = if episodes[0].train {
        "train"
    } else {
        "heldout"
    };
    let history = match history {
        History::Full => "ordered",
        History::Zero => "zero_each_step",
        History::Shuffled => "rotate_past_state_between_eval_episodes",
    };
    let mut total = Metrics::default();
    for (episode, metric) in episodes.iter().zip(metrics) {
        if !episode.train {
            metric.print(model, split, history, 1, Some(episode.stream));
        }
        total.rows += metric.rows;
        total.target += metric.target;
        total.target_square += metric.target_square;
        total.error += metric.error;
        total.error_square += metric.error_square;
    }
    total.print(model, split, history, episodes.len(), None);
}

fn run_screen(episodes: &[Episode], normalization: &Normalization) -> Result<(), ModelError> {
    let training = episodes
        .iter()
        .filter(|episode| episode.train)
        .collect::<Vec<_>>();
    let heldout = episodes
        .iter()
        .filter(|episode| !episode.train)
        .collect::<Vec<_>>();
    for (split, group) in [("train", &training), ("heldout", &heldout)] {
        let positive = group
            .iter()
            .filter(|episode| episode.total_return > 0.0)
            .count();
        let nonzero = group
            .iter()
            .filter(|episode| episode.total_return != 0.0)
            .count();
        println!(
            "{{\"memory_probe\":\"episodes\",\"split\":\"{split}\",\"episodes\":{},\"positive_full_return\":{positive},\"nonzero_full_return\":{nonzero},\"win_labels_available\":false}}",
            group.len()
        );
        let mut mean = Metrics::default();
        for episode in group {
            for target in &episode.returns {
                mean.observe(f64::from(*target), normalization.target_mean)?;
            }
        }
        mean.print("train_mean", split, "none", group.len(), None);
    }
    for recurrent in [false, true] {
        let started = Instant::now();
        let network = ProbeNetwork::new(recurrent)?;
        let name = if recurrent { "gru32" } else { "feedforward108" };
        let parameters = network
            .parameters()
            .iter()
            .map(|variable| variable.elem_count())
            .sum::<usize>();
        assert!(parameters <= 65_000);
        let steps = train(&network, &training, normalization)?;
        let training_ms = started.elapsed().as_millis();
        report(
            name,
            &training,
            &evaluate(&network, &training, normalization, History::Full)?,
            History::Full,
        );
        for history in [History::Full, History::Zero, History::Shuffled] {
            if !recurrent && history != History::Full {
                continue;
            }
            report(
                name,
                &heldout,
                &evaluate(&network, &heldout, normalization, history)?,
                history,
            );
        }
        println!(
            "{{\"memory_probe\":\"fit\",\"model\":\"{name}\",\"parameters\":{parameters},\"steps\":{steps},\"training_ms\":{training_ms},\"total_ms\":{}}}",
            started.elapsed().as_millis()
        );
    }
    Ok(())
}

fn source_hash(model: &PolicyModel) -> Result<String, ModelError> {
    let mut hash = Sha256::new();
    for parameter in model.parameters() {
        for value in parameter.value.flatten_all()?.to_vec1::<f32>()? {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[test]
fn recurrent_prediction_uses_previous_observations_but_feedforward_does_not() {
    let input = Tensor::ones((1, TRUNK_WIDTH), DType::F32, &Device::Cpu).unwrap();
    let current = input.affine(-0.25, 0.0).unwrap();
    let hidden = Tensor::zeros((1, HIDDEN), DType::F32, &Device::Cpu).unwrap();
    for recurrent in [true, false] {
        let network = ProbeNetwork::new(recurrent).unwrap();
        let (_, history) = network.step(&input, &hidden, true).unwrap();
        let (sequence, _) = network.step(&current, &history, false).unwrap();
        let (single, _) = network.step(&current, &hidden, true).unwrap();
        let difference = (sequence - single)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap();
        let difference = difference.to_scalar::<f32>().unwrap();
        if recurrent {
            assert!(difference > 1.0e-6, "GRU must use observable history");
        } else {
            assert_eq!(difference, 0.0, "control must be memoryless");
        }
        let before = network
            .input
            .weight
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (prediction, _) = network.step(&current, &history, false).unwrap();
        sgd(
            &network,
            &prediction
                .affine(1.0, -1.0)
                .unwrap()
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap(),
        )
        .unwrap();
        let after = network
            .input
            .weight
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_ne!(
            before, after,
            "auxiliary loss must backpropagate into its encoder"
        );
    }
}

#[test]
fn window_reset_removes_previous_episode_and_holdout_cannot_change_normalization() {
    let input = Tensor::ones((1, TRUNK_WIDTH), DType::F32, &Device::Cpu).unwrap();
    let zero = Tensor::zeros((1, HIDDEN), DType::F32, &Device::Cpu).unwrap();
    let network = ProbeNetwork::new(true).unwrap();
    let (_, previous) = network.step(&input, &zero, true).unwrap();
    let (reset, _) = network.step(&input, &previous, true).unwrap();
    let (fresh, _) = network.step(&input, &zero, true).unwrap();
    assert_eq!(
        reset.to_vec2::<f32>().unwrap(),
        fresh.to_vec2::<f32>().unwrap()
    );
    let mut episodes = (0..EPISODES).map(Episode::empty).collect::<Vec<_>>();
    assign_split(&mut episodes).unwrap();
    for episode in &mut episodes {
        episode.inputs.push([episode.stream as f32; TRUNK_WIDTH]);
        episode.returns.push(episode.stream as f32);
    }
    episodes[1].inputs.push([1.0; TRUNK_WIDTH]);
    episodes[1].returns.push(1.0);
    let before = Normalization::fit(&episodes).unwrap();
    let group = episodes.iter().take(SEQUENCES).collect::<Vec<_>>();
    let valid = step_batch(&group, 0, &before).unwrap();
    let mixed = step_batch(&group, 1, &before).unwrap();
    let padding = step_batch(&group, 2, &before).unwrap();
    assert_eq!(valid.valid.dtype(), DType::F32);
    assert_eq!(valid.count, SEQUENCES);
    assert_eq!(padding.count, 0);
    let prior = Tensor::ones((SEQUENCES, HIDDEN), DType::F32, &Device::Cpu).unwrap();
    let next = prior.affine(2.0, 0.0).unwrap();
    assert_eq!(mixed.count, 1);
    let preserved = preserve_padding(&prior, &next, &mixed.valid)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(
        preserved,
        vec![
            vec![1.0; HIDDEN],
            vec![2.0; HIDDEN],
            vec![1.0; HIDDEN],
            vec![1.0; HIDDEN]
        ]
    );
    assert_eq!(
        preserve_padding(&prior, &next, &padding.valid)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap(),
        prior.to_vec2::<f32>().unwrap()
    );
    let masked = valid
        .target
        .sqr()
        .unwrap()
        .mul(&padding.valid)
        .unwrap()
        .sum_all()
        .unwrap();
    assert_eq!(
        masked.to_scalar::<f32>().unwrap(),
        0.0,
        "padding cannot contribute loss"
    );
    for episode in episodes.iter_mut().filter(|episode| !episode.train) {
        episode.inputs[0].fill(1.0e6);
        episode.returns[0] = -1.0e6;
    }
    assert_eq!(before, Normalization::fit(&episodes).unwrap());
    assert_eq!(episodes.iter().filter(|episode| episode.train).count(), 32);
    assert_eq!(episodes.iter().filter(|episode| !episode.train).count(), 8);
    for pair in episodes.as_chunks::<2>().0 {
        assert_eq!(pair[0].train, pair[1].train);
    }
    episodes.pop();
    assert_eq!(
        assign_split(&mut episodes).unwrap_err().to_string(),
        ModelError::InvalidModelState("memory probe requires 40 complete episode streams")
            .to_string()
    );
}
