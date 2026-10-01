//! Evaluation players and the opponent pool file.
//!
//! A player is a rule policy (`teacher`, `harass-push`), a rule policy drawing its
//! styled preset per game (`teacher-styled`, `harass-push-styled`), `weights:<dir>` or
//! `average:<dir>,<dir>,...`: the parameter mean of several runtime weights,
//! e.g. the latest history snapshots. Its key names it across runs: the rule
//! label, the weights file SHA-256, or a SHA-256 over the member hashes in order.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{PolicyDevice, PolicyModel, PpoError, ScriptKind, StyledScript, TrainingArtifact};

const POOL_SCHEMA: &str = "drysua-eval-pool/v1";
pub(crate) const MAX_POOL_ENTRIES: usize = 16;
const MAX_AVERAGE_MEMBERS: usize = 16;
const MAX_POOL_BYTES: u64 = 64 * 1024;
const MAX_NAME_BYTES: usize = 64;
const RUNTIME_FILE: &str = "drysua.weights.safetensors";
const MAX_RUNTIME_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EXECUTABLE_BYTES: u64 = 1024 * 1024 * 1024;

/// How a player chooses actions, before any file is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlayerSpec {
    Script(ScriptKind),
    Styled(StyledScript),
    Weights(PathBuf),
    Average(Vec<PathBuf>),
}

impl PlayerSpec {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        let invalid = || {
            format!(
                "player must be a rule policy, weights:<dir> or average:<dir>,<dir>,..., got {value:?}"
            )
        };
        match value.split_once(':') {
            None => ScriptKind::from_label(value)
                .map(Self::Script)
                .or_else(|| StyledScript::from_label(value).map(Self::Styled))
                .ok_or_else(invalid),
            Some(("weights", directory)) if !directory.is_empty() => {
                Ok(Self::Weights(directory.into()))
            }
            Some(("average", members)) => {
                let members: Vec<PathBuf> = members.split(',').map(PathBuf::from).collect();
                if members.len() < 2
                    || members.len() > MAX_AVERAGE_MEMBERS
                    || members.iter().any(|member| member.as_os_str().is_empty())
                {
                    return Err(format!(
                        "average needs 2..={MAX_AVERAGE_MEMBERS} nonempty directories, got {value:?}"
                    ));
                }
                Ok(Self::Average(members))
            }
            Some(_) => Err(invalid()),
        }
    }

    /// The same player with relative directories taken from `base`.
    fn resolved(self, base: &Path) -> Self {
        match self {
            Self::Script(_) | Self::Styled(_) => self,
            Self::Weights(directory) => Self::Weights(base.join(directory)),
            Self::Average(members) => {
                Self::Average(members.into_iter().map(|path| base.join(path)).collect())
            }
        }
    }

    pub(crate) fn label(&self) -> String {
        let join = |members: &[PathBuf]| {
            members
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        match self {
            Self::Script(kind) => kind.label().to_owned(),
            Self::Styled(script) => script.label().to_owned(),
            Self::Weights(directory) => format!("weights:{}", directory.display()),
            Self::Average(members) => format!("average:{}", join(members)),
        }
    }
}

/// Whether training plays against a pool entry; reports show held-out results apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Train,
    HeldOut,
}

impl Role {
    const fn label(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::HeldOut => "held-out",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PoolEntry {
    pub name: String,
    pub player: PlayerSpec,
    pub role: Role,
}

impl PoolEntry {
    pub(crate) fn json(&self, key: &str) -> Value {
        json!({"name": self.name, "player": self.player.label(), "role": self.role.label(), "key": key})
    }
}

/// Names label JSON lines and result files, so they stay short and path-safe.
pub(crate) fn validate_name(name: &str) -> Result<(), String> {
    let allowed = |byte: u8| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-');
    if name.is_empty()
        || name.len() > MAX_NAME_BYTES
        || name.starts_with('.')
        || !name.bytes().all(allowed)
    {
        return Err(format!(
            "player name must be 1..={MAX_NAME_BYTES} of [A-Za-z0-9._-] not starting with '.', got {name:?}"
        ));
    }
    Ok(())
}

/// Reads `{"schema": "drysua-eval-pool/v1", "opponents": [{"name", "player", "role"}]}`;
/// relative directories are relative to the pool file.
pub(crate) fn read_pool(path: &Path) -> Result<Vec<PoolEntry>, PpoError> {
    use std::io::Read;
    let describe = |error: String| PpoError::Model(format!("{}: {error}", path.display()));
    let mut text = String::new();
    std::fs::File::open(path)
        .map_err(|error| describe(error.to_string()))?
        .take(MAX_POOL_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| describe(error.to_string()))?;
    if text.len() as u64 > MAX_POOL_BYTES {
        return Err(describe(format!("larger than {MAX_POOL_BYTES} bytes")));
    }
    let value: Value = serde_json::from_str(&text).map_err(|error| describe(error.to_string()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    parse_pool(&value, base).map_err(describe)
}

fn parse_pool(value: &Value, base: &Path) -> Result<Vec<PoolEntry>, String> {
    let object = value.as_object().ok_or("pool must be an object")?;
    if object.len() != 2 || object.get("schema") != Some(&json!(POOL_SCHEMA)) {
        return Err(format!(
            "pool must hold exactly `schema` ({POOL_SCHEMA}) and `opponents`"
        ));
    }
    let opponents = object
        .get("opponents")
        .and_then(Value::as_array)
        .filter(|opponents| (1..=MAX_POOL_ENTRIES).contains(&opponents.len()))
        .ok_or(format!(
            "opponents must list 1..={MAX_POOL_ENTRIES} entries"
        ))?;
    let mut entries: Vec<PoolEntry> = Vec::with_capacity(opponents.len());
    for opponent in opponents {
        let field = |name: &str| opponent.get(name).and_then(Value::as_str);
        let (Some(name), Some(player), Some(role), Some(3)) = (
            field("name"),
            field("player"),
            field("role"),
            opponent.as_object().map(serde_json::Map::len),
        ) else {
            return Err("every opponent holds exactly string `name`, `player` and `role`".into());
        };
        validate_name(name)?;
        if entries.iter().any(|entry| entry.name == name) {
            return Err(format!("duplicate opponent name {name:?}"));
        }
        let role = match role {
            "train" => Role::Train,
            "held-out" => Role::HeldOut,
            other => return Err(format!("role must be train or held-out, got {other:?}")),
        };
        entries.push(PoolEntry {
            name: name.to_owned(),
            player: PlayerSpec::parse(player)?.resolved(base),
            role,
        });
    }
    Ok(entries)
}

/// How a loaded player acts.
pub(crate) enum Policy {
    Script(ScriptKind),
    Styled(StyledScript),
    Neural(Arc<PolicyModel>),
}

pub(crate) struct Player {
    pub key: String,
    pub policy: Policy,
}

impl Player {
    pub(crate) fn model(&self) -> Option<&Arc<PolicyModel>> {
        match &self.policy {
            Policy::Script(_) | Policy::Styled(_) => None,
            Policy::Neural(model) => Some(model),
        }
    }
}

pub(crate) fn load_player(spec: &PlayerSpec, device: PolicyDevice) -> Result<Player, PpoError> {
    match spec {
        PlayerSpec::Script(kind) => Ok(Player {
            key: format!("script:{}", kind.label()),
            policy: Policy::Script(*kind),
        }),
        PlayerSpec::Styled(script) => Ok(Player {
            key: format!("script:{}", script.label()),
            policy: Policy::Styled(*script),
        }),
        PlayerSpec::Weights(directory) => Ok(Player {
            key: format!("weights:{}", weights_sha256(directory)?),
            policy: Policy::Neural(Arc::new(load_model(directory, device)?)),
        }),
        PlayerSpec::Average(members) => load_average(members, device),
    }
}

/// Averages parameters in `f64` in member order, so one member list is one model.
#[allow(
    clippy::float_arithmetic,
    clippy::cast_possible_truncation,
    reason = "a parameter mean of finite f32 weights"
)]
fn load_average(members: &[PathBuf], device: PolicyDevice) -> Result<Player, PpoError> {
    assert!((2..=MAX_AVERAGE_MEMBERS).contains(&members.len()));
    let mut hashes = Vec::with_capacity(members.len());
    let mut sum: Vec<f64> = Vec::new();
    for member in members {
        hashes.push(weights_sha256(member)?);
        let parameters = load_model(member, PolicyDevice::Cpu)?
            .export_parameters()
            .map_err(super::text_error)?;
        if sum.is_empty() {
            sum = vec![0.0; parameters.len()];
        }
        if sum.len() != parameters.len() {
            return Err(PpoError::InvalidConfig(
                "averaged weights mix side-network layouts",
            ));
        }
        for (total, value) in sum.iter_mut().zip(parameters) {
            *total += f64::from(value);
        }
    }
    let count = members.len() as f64;
    let mean: Vec<f32> = sum
        .into_iter()
        .map(|total| (total / count) as f32)
        .collect();
    let side_networks = crate::SideNetworks::from_parameter_count(mean.len())
        .ok_or(PpoError::InvalidConfig("averaged parameter count"))?;
    let model = PolicyModel::fresh_networks(0, side_networks, device).map_err(super::text_error)?;
    model.import_parameters(&mean).map_err(super::text_error)?;
    Ok(Player {
        key: format!("average:{}", hex(&Sha256::digest(hashes.join(",")))),
        policy: Policy::Neural(Arc::new(model)),
    })
}

fn load_model(directory: &Path, device: PolicyDevice) -> Result<PolicyModel, PpoError> {
    TrainingArtifact::load_runtime_model(directory, device).map_err(|error| {
        PpoError::Model(format!(
            "{}: {error}",
            directory.join(RUNTIME_FILE).display()
        ))
    })
}

fn weights_sha256(directory: &Path) -> Result<String, PpoError> {
    file_sha256(&directory.join(RUNTIME_FILE), MAX_RUNTIME_BYTES)
}

/// Results depend on this exact binary (rules, Teacher, encoding) and on sampling,
/// so only games of one context are paired or rated together.
pub(crate) fn evaluation_context(greedy: bool) -> Result<String, PpoError> {
    let executable = std::env::current_exe()
        .map_err(|error| PpoError::Model(format!("current executable: {error}")))?;
    let digest = file_sha256(&executable, MAX_EXECUTABLE_BYTES)?;
    Ok(format!(
        "exe:{digest}/{}",
        if greedy { "greedy" } else { "sampled" }
    ))
}

fn file_sha256(path: &Path, maximum: u64) -> Result<String, PpoError> {
    use std::io::Read;
    let describe = |error: std::io::Error| PpoError::Model(format!("{}: {error}", path.display()));
    let mut file = std::fs::File::open(path)
        .map_err(describe)?
        .take(maximum + 1);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buffer).map_err(describe)?;
        if read == 0 {
            break;
        }
        total += read as u64;
        hasher.update(&buffer[..read]);
    }
    if total > maximum {
        return Err(PpoError::Model(format!(
            "{} exceeds {maximum} bytes",
            path.display()
        )));
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
