//! Native, read-only checkpoint projection; no model, optimizer or device is constructed.
use super::*;
use serde_json::{Value, json};
use std::io::Write;

#[path = "checkpoint_inspection_history.rs"]
mod history;
#[path = "checkpoint_inspection_io.rs"]
mod io;
#[path = "checkpoint_inspection_json.rs"]
mod projection;
#[cfg(test)]
#[path = "tests/checkpoint_inspection.rs"]
mod tests;

use io::Directory;

const SCHEMA: &str = "drysua-checkpoint-inspection/v1";
const MAX_JSON_BYTES: usize = 4 * 1024 * 1024;
const MAX_SNAPSHOTS: u64 = 10_000;
const MAX_FILES: usize = MAX_SNAPSHOTS as usize + 4;
const CHANGED: &str = "checkpoint changed during inspection";

impl TrainingArtifact {
    /// Validates and describes checkpoint files without restoring a model or writing files.
    pub fn inspect(directory: &Path) -> Result<Vec<u8>, CheckpointError> {
        checkpoint_inspect(directory)
    }
}

/// Returns one bounded JSON document and newline, or an error without output or writes.
pub fn checkpoint_inspect(directory: &Path) -> Result<Vec<u8>, CheckpointError> {
    inspect_with_hook(directory, || {})
}

/// Describes the exact current build's checkpoint identities without filesystem or device I/O.
pub fn checkpoint_inspection_contract() -> Result<Vec<u8>, CheckpointError> {
    json_bytes(projection::contract())
}

fn inspect_with_hook(
    directory: &Path,
    after_inventory: impl FnOnce(),
) -> Result<Vec<u8>, CheckpointError> {
    let root = Directory::open(directory)?;
    let manifest = read_manifest(&root)?;
    let inspected = inspect_manifest(&root, &manifest);
    after_inventory();
    // Re-read even after a payload error: a new commit is not a corrupt old commit.
    let current = read_manifest(&root).map_err(|_| CheckpointError::InvalidManifest(CHANGED))?;
    if current != manifest {
        return Err(CheckpointError::InvalidManifest(CHANGED));
    }
    root.check_path(directory)?;
    json_bytes(inspected?)
}

fn inspect_manifest(root: &Directory, manifest: &[u8]) -> Result<Value, CheckpointError> {
    let artifact = decode_manifest(manifest)?;
    let plan = history::plan(&artifact)?;
    let mut files = Vec::with_capacity(plan.snapshot_count as usize + 4);
    add_file(&mut files, CHECKPOINT_META_FILE, manifest)?;
    let (artifact, tensor_path) = load_payload(root, artifact, &mut files)?;
    let runtime = inspect_runtime(root, &artifact, &mut files)?;
    let history = history::verify(root, &artifact, &plan, &mut files)?;
    Ok(json!({
        "schema": SCHEMA, "kind": plan.kind,
        "identity": {"manifest_sha256": hex(&sha256(manifest)), "tensor_sha256": hex(&artifact.tensor_hash),
            "runtime_sha256": runtime.hash, "scope_sha256": scope_hash(&artifact)?},
        "model": projection::model(), "checkpoint": projection::checkpoint(),
        "progress": projection::progress(&artifact, plan.games), "run": projection::run(&artifact.run),
        "ppo": projection::ppo(artifact.config), "adaptive": projection::adaptive(&artifact.progress),
        "history": history, "runtime_status": runtime.status, "runtime_matches_model": runtime.matches,
        "recovery_required": !runtime.matches,
        "sources": {"manifest": CHECKPOINT_META_FILE, "tensor": tensor_path,
            "runtime": runtime.hash.as_ref().map(|_| RUNTIME_TENSOR_FILE)},
        "files": files,
    }))
}

fn load_payload(
    root: &Directory,
    mut artifact: TrainingArtifact,
    files: &mut Vec<Value>,
) -> Result<(TrainingArtifact, String), CheckpointError> {
    let generation = tensor_generation_path(Path::new(""), artifact.tensor_hash);
    let generation = generation
        .to_str()
        .ok_or(CheckpointError::InvalidManifest("inspection tensor name"))?;
    let tensors = root
        .read(generation, MAX_TRAINING_TENSOR_BYTES)?
        .ok_or_else(|| CheckpointError::Io(format!("missing checkpoint artifact: {generation}")))?;
    let path = generation.to_owned();
    if sha256(&tensors) != artifact.tensor_hash {
        return Err(CheckpointError::TensorHashMismatch);
    }
    // Reuse the native codecs and full artifact validation, not a second binary parser.
    // Unlike load's pathname reopens, every inspection read remains descriptor-anchored.
    let decoded = decode_training_tensors(&tensors)?;
    artifact.parameters = decoded.parameters;
    artifact.optimizer.first_moment = decoded.first_moment;
    artifact.optimizer.second_moment = decoded.second_moment;
    artifact.collection = decoded.collection;
    artifact.validate()?;
    add_file(files, &path, &tensors)?;
    Ok((artifact, path))
}

struct RuntimeInspection {
    status: &'static str,
    matches: bool,
    hash: Option<String>,
}

fn inspect_runtime(
    root: &Directory,
    artifact: &TrainingArtifact,
    files: &mut Vec<Value>,
) -> Result<RuntimeInspection, CheckpointError> {
    let Some(bytes) = root.read(RUNTIME_TENSOR_FILE, MAX_RUNTIME_TENSOR_BYTES)? else {
        return Ok(RuntimeInspection {
            status: "missing",
            matches: false,
            hash: None,
        });
    };
    let hash = Some(hex(&sha256(&bytes)));
    let parameters = match decode_runtime_parameters(&bytes) {
        Ok(parameters) => parameters,
        Err(CheckpointError::SchemaMismatch) => {
            return Ok(RuntimeInspection {
                status: "mismatch",
                matches: false,
                hash,
            });
        }
        Err(error) => return Err(error),
    };
    let matches = parameters.len() == artifact.parameters.len()
        && parameters
            .iter()
            .zip(&artifact.parameters)
            .all(|(left, right)| left.to_bits() == right.to_bits());
    if matches {
        add_file(files, RUNTIME_TENSOR_FILE, &bytes)?;
    }
    Ok(RuntimeInspection {
        status: if matches { "matched" } else { "mismatch" },
        matches,
        hash,
    })
}

fn read_manifest(root: &Directory) -> Result<Vec<u8>, CheckpointError> {
    root.read(CHECKPOINT_META_FILE, MAX_META_BYTES)?
        .ok_or_else(|| {
            CheckpointError::Io(format!(
                "missing checkpoint artifact: {CHECKPOINT_META_FILE}"
            ))
        })
}

fn scope_hash(artifact: &TrainingArtifact) -> Result<String, CheckpointError> {
    let mut writer = ManifestWriter::default();
    writer
        .bytes
        .extend_from_slice(b"drysua-checkpoint-inspection-scope/v1\0");
    writer.u32(CHECKPOINT_SCHEMA_VERSION);
    writer.u64(CHECKPOINT_SCHEMA_HASH);
    encode_schema(&mut writer);
    encode_run(&mut writer, &artifact.run)?;
    encode_config(&mut writer, artifact.config)?;
    assert!(writer.bytes.len() <= MAX_META_BYTES as usize);
    Ok(hex(&sha256(&writer.bytes)))
}

fn add_file(files: &mut Vec<Value>, path: &str, bytes: &[u8]) -> Result<(), CheckpointError> {
    if files.len() >= MAX_FILES || path.len() > 128 {
        return Err(CheckpointError::InvalidManifest(
            "inspection file inventory",
        ));
    }
    let parts = path.split('/').collect::<Vec<_>>();
    if !(1..=2).contains(&parts.len())
        || (parts.len() == 2 && parts[0] != "domain-randomization")
        || parts.iter().any(|part| {
            part.is_empty()
                || *part == "."
                || *part == ".."
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
        })
    {
        return Err(CheckpointError::InvalidManifest(
            "inspection file inventory",
        ));
    }
    files.push(json!({"path": path, "size": bytes.len(), "sha256": hex(&sha256(bytes))}));
    Ok(())
}

fn hex(hash: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write as _;
        write!(&mut text, "{byte:02x}").expect("String write");
    }
    assert_eq!(text.len(), 64);
    text
}

fn json_bytes(value: Value) -> Result<Vec<u8>, CheckpointError> {
    let mut output = JsonOutput(Vec::with_capacity(8192));
    serde_json::to_writer(&mut output, &value)
        .map_err(|error| CheckpointError::Io(format!("inspection JSON: {error}")))?;
    output.write_all(b"\n")?;
    assert!(output.0.len() <= MAX_JSON_BYTES);
    Ok(output.0)
}

struct JsonOutput(Vec<u8>);

impl Write for JsonOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_JSON_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "inspection JSON exceeds 4 MiB",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
