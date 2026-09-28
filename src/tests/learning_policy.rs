use crate::{MODEL_PARAMETER_COUNT, PpoError, StateTracker};
use bota_proto::Team;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(super) const SHARPENED_PARAMETERS: usize = 112_963;
const _: () = assert!(MODEL_PARAMETER_COUNT - SHARPENED_PARAMETERS == 1_587_057);
const SHARPENED_TENSORS: [&str; 24] = [
    "kind.weight",
    "kind.bias",
    "controlled.weight",
    "controlled.bias",
    "ability_head.weight",
    "ability_head.bias",
    "item_head.weight",
    "item_head.bias",
    "swap_head.weight",
    "swap_head.bias",
    "learn_head.weight",
    "learn_head.bias",
    "shop_head.weight",
    "shop_head.bias",
    "loot_head.weight",
    "loot_head.bias",
    "target_mode.weight",
    "target_mode.bias",
    "put_mode.weight",
    "put_mode.bias",
    "entity_query.weight",
    "entity_query.bias",
    "point_query.weight",
    "point_query.bias",
];

#[cfg(test)]
mod sharpen_tests {
    use super::*;

    #[test]
    fn sharpen_four_scales_exact_actor_heads_and_preserves_all_other_bits() {
        let model = crate::PolicyModel::fresh(714).unwrap();
        let parameters: Vec<_> = (0..MODEL_PARAMETER_COUNT)
            .map(|index| [0.0, -0.0, 0.125, -0.25][index % 4])
            .collect();
        model.import_parameters(&parameters).unwrap();
        let identity = model.policy_identity().unwrap();
        let output = sharpen_four(&model).unwrap();
        let mut offset = 0;
        let mut affected = 0;
        let schema = model.parameter_schema().unwrap();
        assert_eq!(schema.len(), 62);
        for (name, shape) in schema {
            let end = offset + shape.iter().product::<usize>();
            let selected = SHARPENED_TENSORS.contains(&name);
            for index in offset..end {
                let expected = if selected {
                    parameters[index] * 4.0
                } else {
                    parameters[index]
                };
                assert_eq!(
                    output[index].to_bits(),
                    expected.to_bits(),
                    "{name}:{index}"
                );
            }
            if selected {
                affected += end - offset;
            }
            offset = end;
        }
        assert_eq!(affected, 112_963);
        assert_eq!(offset - affected, 1_587_057);
        assert_eq!(model.policy_identity().unwrap(), identity);
        assert!(
            model
                .export_parameters()
                .unwrap()
                .iter()
                .zip(&parameters)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn sharpen_four_rejects_overflow_without_mutating_source_model() {
        let model = crate::PolicyModel::fresh(715).unwrap();
        let mut parameters = model.export_parameters().unwrap();
        let mut offset = 0;
        for (name, shape) in model.parameter_schema().unwrap() {
            if name == "point_query.bias" {
                parameters[offset] = f32::MAX;
            }
            offset += shape.iter().product::<usize>();
        }
        model.import_parameters(&parameters).unwrap();
        let identity = model.policy_identity().unwrap();
        let error = sharpen_four(&model).unwrap_err();
        assert_eq!(error.to_string(), "PPO sharpened parameter is non-finite");
        assert_eq!(model.policy_identity().unwrap(), identity);
        assert_eq!(model.export_parameters().unwrap(), parameters);
    }
}

/// Fixed post-training calibration, not a trainable temperature or optimizer step.
/// Mathematically scales conditional logits by four; f32 greedy parity must be checked.
pub(super) fn sharpen_four(model: &crate::PolicyModel) -> Result<Vec<f32>, PpoError> {
    if crate::MODEL_SCHEMA_VERSION != 24 {
        return Err(PpoError::InvalidConfig("sharpen requires model schema 24"));
    }
    let schema = model
        .parameter_schema()
        .map_err(|error| PpoError::Model(error.to_string()))?;
    if schema.len() != 62 {
        return Err(PpoError::InvalidConfig("sharpen tensor count"));
    }
    let mut output = model
        .export_parameters()
        .map_err(|error| PpoError::Model(error.to_string()))?;
    if output.len() != MODEL_PARAMETER_COUNT || output.iter().any(|value| !value.is_finite()) {
        return Err(PpoError::InvalidTransition("sharpen source parameters"));
    }
    let mut seen = [false; 24];
    let mut offset = 0usize;
    let mut affected = 0usize;
    for (name, shape) in schema {
        let count = shape
            .iter()
            .try_fold(1usize, |count, size| count.checked_mul(*size))
            .ok_or(PpoError::InvalidTransition("sharpen tensor shape"))?;
        let end = offset
            .checked_add(count)
            .filter(|end| *end <= output.len())
            .ok_or(PpoError::InvalidTransition("sharpen tensor range"))?;
        if let Some(index) = SHARPENED_TENSORS
            .iter()
            .position(|expected| *expected == name)
        {
            if seen[index] {
                return Err(PpoError::InvalidTransition("duplicate sharpen tensor"));
            }
            seen[index] = true;
            affected += count;
            for value in &mut output[offset..end] {
                *value *= 4.0;
                if !value.is_finite() {
                    return Err(PpoError::NonFinite("sharpened parameter"));
                }
            }
        }
        offset = end;
    }
    if offset != MODEL_PARAMETER_COUNT || affected != SHARPENED_PARAMETERS || seen.contains(&false)
    {
        return Err(PpoError::InvalidTransition("sharpen actor tensor contract"));
    }
    Ok(output)
}

pub(super) const ROUTING: &str = concat!(
    "bota-drysua-team-ensemble/v1;selector=observed_own_player_team;",
    "selection=once_at_game_start;Radiant=0;Dire=1;Neutral=error;inputs=own_team_only;",
    "identity=sha256(descriptor_utf8_nul,action_feature_model_version_le32_hash_le64,",
    "radiant_lower_hex64,dire_lower_hex64)"
);

pub(super) fn team_index(team: Team) -> Result<usize, PpoError> {
    match team {
        Team::Radiant => Ok(0),
        Team::Dire => Ok(1),
        Team::Neutral => Err(PpoError::InvalidConfig("routing requires Radiant or Dire")),
    }
}

pub(super) fn observed_team(tracker: &StateTracker) -> Result<usize, PpoError> {
    let player = tracker.own_player().ok_or(PpoError::InvalidConfig(
        "routing requires observed own player",
    ))?;
    if player.team != tracker.team() {
        return Err(PpoError::InvalidConfig(
            "observed own team differs from tracker team",
        ));
    }
    team_index(player.team)
}

pub(super) fn ensemble_id(radiant_hash: &str, dire_hash: &str) -> Result<String, PpoError> {
    let radiant = normalized_hash(radiant_hash, "radiant runtime SHA256 must be 64 hex digits")?;
    let dire = normalized_hash(dire_hash, "dire runtime SHA256 must be 64 hex digits")?;
    let mut digest = Sha256::new();
    digest.update(ROUTING.as_bytes());
    digest.update([0]);
    for (version, hash) in [
        (crate::ACTION_SCHEMA_VERSION, crate::ACTION_SCHEMA_HASH),
        (crate::FEATURE_SCHEMA_VERSION, crate::FEATURE_SCHEMA_HASH),
        (crate::MODEL_SCHEMA_VERSION, crate::MODEL_SCHEMA_HASH),
    ] {
        digest.update(version.to_le_bytes());
        digest.update(hash.to_le_bytes());
    }
    digest.update(radiant.as_bytes());
    digest.update(dire.as_bytes());
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub(super) fn validate_declaration(
    declaration: &Value,
    identity: &str,
    experts: Option<(&str, &str)>,
    offset: u64,
    mode: &str,
) -> Result<(), PpoError> {
    let games = declaration_games(declaration, mode)?;
    if !offset.is_multiple_of(20) || offset > games / 2 - 20 {
        return Err(PpoError::InvalidConfig(
            "final offset must be a complete aligned 20-pair chunk",
        ));
    }
    let candidate = declared_candidate(declaration)?;
    let control = declaration_hash(declaration, "control_runtime_sha256")?;
    // Validate the entire declaration before admitting even the control runtime.
    let identity = normalized_hash(identity, "evaluation identity must be 64 hex digits")?;
    if let Some((radiant, dire)) = experts {
        let actual = (
            normalized_hash(radiant, "radiant runtime SHA256 must be 64 hex digits")?,
            normalized_hash(dire, "dire runtime SHA256 must be 64 hex digits")?,
        );
        if identity != candidate.identity || candidate.experts.as_ref() != Some(&actual) {
            return Err(PpoError::InvalidConfig(
                "ensemble identity or experts differ from declaration",
            ));
        }
    } else if identity != control && (candidate.experts.is_some() || identity != candidate.identity)
    {
        return Err(PpoError::InvalidConfig("single runtime is not preselected"));
    }
    Ok(())
}

fn declaration_games(declaration: &Value, mode: &str) -> Result<u64, PpoError> {
    let object = declaration
        .as_object()
        .ok_or(PpoError::InvalidConfig("declaration must be an object"))?;
    if object.keys().any(|field| {
        !matches!(
            field.as_str(),
            "schema"
                | "set"
                | "seed"
                | "purpose"
                | "games_per_model"
                | "candidate_mode"
                | "control_mode"
                | "control_runtime_sha256"
                | "candidate_runtime_sha256"
                | "candidate_radiant_runtime_sha256"
                | "candidate_dire_runtime_sha256"
                | "candidate_routing"
                | "candidate_policy_sha256"
        )
    }) {
        return Err(PpoError::InvalidConfig("unknown declaration field"));
    }
    for (field, expected) in [
        ("schema", "drysua-final-evaluation/v1"),
        ("set", "autonomous-final"),
    ] {
        if declaration[field].as_str() != Some(expected) {
            return Err(PpoError::InvalidConfig(field));
        }
    }
    if declaration["seed"].as_u64() != Some(2026092803) {
        return Err(PpoError::InvalidConfig("seed"));
    }
    let (required_mode, required_games): (&str, &[u64]) = match declaration["purpose"].as_str() {
        Some("primary") => ("greedy", &[200]),
        Some("diagnostic") => ("sampled", &[80]),
        _ => return Err(PpoError::InvalidConfig("purpose")),
    };
    for field in ["candidate_mode", "control_mode"] {
        if declaration[field].as_str() != Some(required_mode) {
            return Err(PpoError::InvalidConfig(field));
        }
    }
    if mode != required_mode {
        return Err(PpoError::InvalidConfig(
            "evaluation mode differs from declaration",
        ));
    }
    declaration["games_per_model"]
        .as_u64()
        .filter(|games| required_games.contains(games))
        .ok_or(PpoError::InvalidConfig("games_per_model"))
}

struct DeclaredCandidate {
    identity: String,
    experts: Option<(String, String)>,
}

fn declared_candidate(declaration: &Value) -> Result<DeclaredCandidate, PpoError> {
    let ensemble = [
        "candidate_radiant_runtime_sha256",
        "candidate_dire_runtime_sha256",
        "candidate_routing",
        "candidate_policy_sha256",
    ]
    .iter()
    .any(|field| declaration.get(*field).is_some());
    if ensemble && declaration.get("candidate_runtime_sha256").is_some() {
        return Err(PpoError::InvalidConfig(
            "mixed single and ensemble candidate",
        ));
    }
    let (identity, experts) = if ensemble {
        let radiant = declaration_hash(declaration, "candidate_radiant_runtime_sha256")?;
        let dire = declaration_hash(declaration, "candidate_dire_runtime_sha256")?;
        if declaration["candidate_routing"].as_str() != Some(ROUTING) {
            return Err(PpoError::InvalidConfig("candidate_routing"));
        }
        let candidate = declaration_hash(declaration, "candidate_policy_sha256")?;
        if candidate != ensemble_id(&radiant, &dire)? {
            return Err(PpoError::InvalidConfig(
                "candidate policy SHA256 does not bind ordered experts",
            ));
        }
        (candidate, Some((radiant, dire)))
    } else {
        (
            declaration_hash(declaration, "candidate_runtime_sha256")?,
            None,
        )
    };
    Ok(DeclaredCandidate { identity, experts })
}

fn declaration_hash(declaration: &Value, field: &'static str) -> Result<String, PpoError> {
    let value = declaration[field]
        .as_str()
        .ok_or(PpoError::InvalidConfig(field))?;
    normalized_hash(value, field)
}

fn normalized_hash(value: &str, field: &'static str) -> Result<String, PpoError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PpoError::InvalidConfig(field));
    }
    Ok(value.to_ascii_lowercase())
}

/// Radiant-weighted interpolation: alpha * Radiant + (1 - alpha) * Dire.
/// Alpha zero copies Dire; alpha one copies Radiant, preserving endpoint bits.
pub(super) fn blend_parameters(
    radiant: &[f32],
    dire: &[f32],
    alpha: f32,
) -> Result<Vec<f32>, PpoError> {
    if radiant.len() != MODEL_PARAMETER_COUNT || dire.len() != MODEL_PARAMETER_COUNT {
        return Err(PpoError::InvalidConfig(
            "blend requires exact MODEL_PARAMETER_COUNT",
        ));
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(PpoError::InvalidConfig(
            "blend alpha must be finite in 0..=1",
        ));
    }
    if !radiant.iter().chain(dire).all(|value| value.is_finite()) {
        return Err(PpoError::InvalidConfig("blend parameters must be finite"));
    }
    // Arithmetic would erase signed zero even when the endpoint is unchanged.
    if alpha == 0.0 {
        return Ok(dire.to_vec());
    }
    if alpha == 1.0 {
        return Ok(radiant.to_vec());
    }
    let alpha = f64::from(alpha);
    radiant
        .iter()
        .zip(dire)
        .map(|(&radiant, &dire)| {
            let value = (alpha * f64::from(radiant) + (1.0 - alpha) * f64::from(dire)) as f32;
            if !value.is_finite() {
                return Err(PpoError::InvalidConfig("blended parameters must be finite"));
            }
            Ok(value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ROUTING, blend_parameters, ensemble_id, observed_team, team_index, validate_declaration,
    };
    use crate::{MODEL_PARAMETER_COUNT, PpoError, SHADOW_FIEND, StateTracker};
    use bota_proto::{MapId, MatchInfo, Pick, PlayerView, SlotId, Team, TickMode, WorldView};
    use serde_json::{Value, json};

    fn rejects<T: std::fmt::Debug>(result: Result<T, PpoError>, field: &str) {
        assert_eq!(
            result.unwrap_err().to_string(),
            format!("invalid PPO config field: {field}")
        );
    }

    fn match_info(team: Team) -> MatchInfo {
        MatchInfo {
            match_id: 7,
            map: MapId(0),
            tick_rate: 30,
            pregame_ticks: 0,
            trees: vec![],
            terrain_cells: 1,
            terrain_rle: vec![(1, 0x80)],
            opaque_cells: vec![],
            mode: TickMode::Lockstep,
            shop: vec![],
            picks: vec![
                Pick {
                    slot: SlotId(1),
                    team: if team == Team::Radiant {
                        Team::Dire
                    } else {
                        Team::Radiant
                    },
                    hero: SHADOW_FIEND,
                },
                Pick {
                    slot: SlotId(0),
                    team,
                    hero: SHADOW_FIEND,
                },
            ],
        }
    }

    #[test]
    fn routing_uses_own_pick_team_not_slot_roster_order_opponent_or_statistics() {
        for team in [Team::Radiant, Team::Dire] {
            let info = match_info(team);
            for slot in [SlotId(0), SlotId(1)] {
                let mut tracker = StateTracker::new(slot, &info).unwrap();
                let players = [1, 0]
                    .map(|index| {
                        let pick = &info.picks[index];
                        PlayerView {
                            slot: pick.slot,
                            team: pick.team,
                            hero: pick.hero,
                            unit: None,
                            level: 1,
                            xp: 0,
                            gold: Some(0),
                            stash: Some(vec![None; 6]),
                            kit: None,
                            kills: 0,
                            deaths: 0,
                            assists: 0,
                            last_hits: 0,
                            denies: 0,
                            respawn_left: 10,
                        }
                    })
                    .to_vec();
                let mut view = WorldView {
                    tick: 1,
                    viewer: Some(tracker.team()),
                    players,
                    units: vec![],
                    projectiles: vec![],
                    felled_trees: vec![],
                    planted_trees: vec![],
                    loot: vec![],
                };
                let expected = usize::from(tracker.team() == Team::Dire);
                tracker.observe_snapshot(&view).unwrap();
                assert_eq!(observed_team(&tracker).unwrap(), expected);
                let mut same_team_info = info.clone();
                for pick in &mut same_team_info.picks {
                    pick.team = tracker.team();
                }
                let mut same_team_view = view.clone();
                for player in &mut same_team_view.players {
                    player.team = tracker.team();
                }
                let mut same_team = StateTracker::new(slot, &same_team_info).unwrap();
                same_team.observe_snapshot(&same_team_view).unwrap();
                assert_eq!(observed_team(&same_team).unwrap(), expected);
                view.tick = 2;
                for player in &mut view.players {
                    player.kills = 91;
                    player.last_hits = 137;
                }
                tracker.observe_snapshot(&view).unwrap();
                assert_eq!(observed_team(&tracker).unwrap(), expected);
            }
        }
    }

    #[test]
    fn routing_rejects_neutral_and_unobserved_own_player() {
        assert_eq!(team_index(Team::Radiant).unwrap(), 0);
        assert_eq!(team_index(Team::Dire).unwrap(), 1);
        rejects(
            team_index(Team::Neutral),
            "routing requires Radiant or Dire",
        );
        let tracker = StateTracker::new(SlotId(0), &match_info(Team::Radiant)).unwrap();
        rejects(
            observed_team(&tracker),
            "routing requires observed own player",
        );
        assert_eq!(
            StateTracker::new(SlotId(0), &match_info(Team::Neutral))
                .err()
                .unwrap()
                .to_string(),
            "own slot has non-playable team Neutral"
        );
    }

    #[test]
    fn ensemble_identity_normalizes_hex_but_preserves_expert_order_and_all_digest_bits() {
        let radiant = "ab".repeat(32);
        let dire = "cd".repeat(32);
        let identity = ensemble_id(&radiant, &dire).unwrap();
        assert_eq!(identity.len(), 64);
        assert!(
            identity
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
        assert_eq!(
            identity,
            ensemble_id(&radiant.to_uppercase(), &dire.to_uppercase()).unwrap()
        );
        assert_ne!(identity, ensemble_id(&dire, &radiant).unwrap());
        for invalid in [
            "a".repeat(63),
            "a".repeat(65),
            "g".repeat(64),
            "é".repeat(32),
        ] {
            rejects(
                ensemble_id(&invalid, &dire),
                "radiant runtime SHA256 must be 64 hex digits",
            );
            rejects(
                ensemble_id(&radiant, &invalid),
                "dire runtime SHA256 must be 64 hex digits",
            );
        }
    }

    fn declaration(ensemble: bool) -> Value {
        let mut value = json!({"schema":"drysua-final-evaluation/v1",
            "set":"autonomous-final", "seed":2026092803_u64, "purpose":"primary",
            "games_per_model":200, "candidate_mode":"greedy", "control_mode":"greedy",
            "control_runtime_sha256":"ef".repeat(32)});
        if ensemble {
            value["candidate_radiant_runtime_sha256"] = json!("ab".repeat(32));
            value["candidate_dire_runtime_sha256"] = json!("cd".repeat(32));
            value["candidate_routing"] = json!(ROUTING);
            value["candidate_policy_sha256"] =
                json!(ensemble_id(&"ab".repeat(32), &"cd".repeat(32)).unwrap());
        } else {
            value["candidate_runtime_sha256"] = json!("ab".repeat(32));
        }
        value
    }

    fn rejects_before_identity_admission(value: &Value, ensemble: bool, mode: &str, field: &str) {
        let radiant = "ab".repeat(32);
        let dire = "cd".repeat(32);
        let identity = if ensemble {
            ensemble_id(&radiant, &dire).unwrap()
        } else {
            radiant.clone()
        };
        rejects(
            validate_declaration(
                value,
                &identity,
                ensemble.then_some((radiant.as_str(), dire.as_str())),
                0,
                mode,
            ),
            field,
        );
        for identity in ["ef".repeat(32), "invalid identity".to_owned()] {
            rejects(validate_declaration(value, &identity, None, 0, mode), field);
        }
    }

    #[test]
    fn declarations_reject_nonobjects_before_identity_admission() {
        for value in [
            Value::Null,
            json!([]),
            json!(true),
            json!(1),
            json!("manifest"),
        ] {
            for ensemble in [false, true] {
                rejects_before_identity_admission(
                    &value,
                    ensemble,
                    "greedy",
                    "declaration must be an object",
                );
            }
        }
    }

    #[test]
    fn declarations_reject_unknown_fields_including_null_future_runtime() {
        for ensemble in [false, true] {
            for field in [
                "candidate_future_runtime_sha256",
                "metadata",
                "chronology",
                "source",
                "hashledger",
            ] {
                for replacement in [Value::Null, json!({}), json!("extra")] {
                    let mut value = declaration(ensemble);
                    value[field] = replacement;
                    rejects_before_identity_admission(
                        &value,
                        ensemble,
                        "greedy",
                        "unknown declaration field",
                    );
                }
            }
        }
    }

    #[test]
    fn declarations_require_nonnull_typed_metadata_and_modes_for_both_variants() {
        for ensemble in [false, true] {
            for field in [
                "schema",
                "set",
                "seed",
                "purpose",
                "games_per_model",
                "candidate_mode",
                "control_mode",
            ] {
                for replacement in [
                    None,
                    Some(Value::Null),
                    Some(json!(true)),
                    Some(json!([])),
                    Some(json!({})),
                ] {
                    let mut value = declaration(ensemble);
                    value.as_object_mut().unwrap().remove(field);
                    if let Some(replacement) = replacement {
                        value[field] = replacement;
                    }
                    rejects_before_identity_admission(&value, ensemble, "greedy", field);
                }
            }
        }
    }

    #[test]
    fn declarations_reject_schema_scope_seed_and_purpose_mismatches() {
        for ensemble in [false, true] {
            for (field, replacement) in [
                ("schema", json!("drysua-final-evaluation/v2")),
                ("schema", json!(1)),
                ("set", json!("autonomous-validation")),
                ("seed", json!(2026092802_u64)),
                ("seed", json!("2026092803")),
                ("seed", json!(2026092803.0)),
                ("seed", json!(-1)),
                ("seed", json!(u64::MAX)),
                ("purpose", json!("evaluation")),
                ("purpose", json!("PRIMARY")),
            ] {
                let mut value = declaration(ensemble);
                value[field] = replacement;
                rejects_before_identity_admission(&value, ensemble, "greedy", field);
            }
        }
    }

    #[test]
    fn declarations_reject_invalid_or_mixed_modes_before_accepting_control() {
        for ensemble in [false, true] {
            for (purpose, games, mode, other) in [
                ("primary", 200, "greedy", "sampled"),
                ("diagnostic", 80, "sampled", "greedy"),
            ] {
                let mut original = declaration(ensemble);
                original["purpose"] = json!(purpose);
                original["games_per_model"] = json!(games);
                original["candidate_mode"] = json!(mode);
                original["control_mode"] = json!(mode);
                for field in ["candidate_mode", "control_mode"] {
                    for replacement in [
                        None,
                        Some(Value::Null),
                        Some(json!(other)),
                        Some(json!("GREEDY")),
                        Some(json!("")),
                        Some(json!(0)),
                    ] {
                        let mut value = original.clone();
                        value.as_object_mut().unwrap().remove(field);
                        if let Some(replacement) = replacement {
                            value[field] = replacement;
                        }
                        rejects_before_identity_admission(&value, ensemble, mode, field);
                    }
                }
                original["candidate_mode"] = json!(other);
                original["control_mode"] = json!(other);
                rejects_before_identity_admission(&original, ensemble, other, "candidate_mode");
            }
        }
    }

    #[test]
    fn declarations_reject_legacy_single_candidate_without_declared_modes() {
        let mut value = declaration(false);
        value.as_object_mut().unwrap().remove("candidate_mode");
        value.as_object_mut().unwrap().remove("control_mode");
        rejects_before_identity_admission(&value, false, "greedy", "candidate_mode");
    }

    #[test]
    fn declarations_reject_actual_mode_mismatch_for_candidate_and_control() {
        for ensemble in [false, true] {
            for (purpose, games, mode, other) in [
                ("primary", 200, "greedy", "sampled"),
                ("diagnostic", 80, "sampled", "greedy"),
            ] {
                let mut value = declaration(ensemble);
                value["purpose"] = json!(purpose);
                value["games_per_model"] = json!(games);
                value["candidate_mode"] = json!(mode);
                value["control_mode"] = json!(mode);
                for actual in [other, "", "GREEDY", "unknown"] {
                    rejects_before_identity_admission(
                        &value,
                        ensemble,
                        actual,
                        "evaluation mode differs from declaration",
                    );
                }
            }
        }
    }

    #[test]
    fn declarations_reject_game_counts_outside_purpose_contract() {
        for ensemble in [false, true] {
            for (purpose, mode, invalid_games) in [
                ("primary", "greedy", [0, 40, 80, 199, 201, 401]),
                ("diagnostic", "sampled", [0, 40, 79, 81, 200, 400]),
            ] {
                let mut value = declaration(ensemble);
                value["purpose"] = json!(purpose);
                value["candidate_mode"] = json!(mode);
                value["control_mode"] = json!(mode);
                for games in invalid_games.map(|games| json!(games)).into_iter().chain([
                    json!("200"),
                    json!(200.0),
                    json!(80.0),
                    json!(-1),
                    json!(u64::MAX),
                ]) {
                    value["games_per_model"] = games;
                    rejects_before_identity_admission(&value, ensemble, mode, "games_per_model");
                }
            }
        }
    }

    #[test]
    fn diagnostic_sampled_80_accepts_offsets_zero_and_twenty_but_rejects_forty() {
        let radiant = "ab".repeat(32);
        let dire = "cd".repeat(32);
        let control = "ef".repeat(32);
        for ensemble in [false, true] {
            let mut value = declaration(ensemble);
            value["purpose"] = json!("diagnostic");
            value["games_per_model"] = json!(80);
            value["candidate_mode"] = json!("sampled");
            value["control_mode"] = json!("sampled");
            let identity = if ensemble {
                ensemble_id(&radiant, &dire).unwrap()
            } else {
                radiant.clone()
            };
            for (identity, experts) in [
                (
                    identity.as_str(),
                    ensemble.then_some((radiant.as_str(), dire.as_str())),
                ),
                (control.as_str(), None),
            ] {
                for offset in [0, 20] {
                    validate_declaration(&value, identity, experts, offset, "sampled").unwrap();
                }
                for offset in [1, 40, u64::MAX] {
                    rejects(
                        validate_declaration(&value, identity, experts, offset, "sampled"),
                        "final offset must be a complete aligned 20-pair chunk",
                    );
                }
                rejects(
                    validate_declaration(&value, identity, experts, 0, "greedy"),
                    "evaluation mode differs from declaration",
                );
            }
        }
    }

    #[test]
    fn primary_declaration_rejects_400_games_for_candidate_and_control() {
        for ensemble in [false, true] {
            let mut value = declaration(ensemble);
            value["games_per_model"] = json!(400);
            rejects_before_identity_admission(&value, ensemble, "greedy", "games_per_model");
        }
    }

    #[test]
    fn declarations_accept_legacy_candidate_control_and_ordered_experts_at_chunk_boundaries() {
        let radiant = "ab".repeat(32);
        let dire = "cd".repeat(32);
        for ensemble in [false, true] {
            let mut value = declaration(ensemble);
            let identity = if ensemble {
                ensemble_id(&radiant, &dire).unwrap()
            } else {
                radiant.clone()
            };
            value["games_per_model"] = json!(200);
            for offset in [0, 80] {
                validate_declaration(
                    &value,
                    &identity,
                    ensemble.then_some((radiant.as_str(), dire.as_str())),
                    offset,
                    "greedy",
                )
                .unwrap();
                validate_declaration(&value, &"EF".repeat(32), None, offset, "greedy").unwrap();
            }
            for offset in [1, 100, u64::MAX] {
                rejects(
                    validate_declaration(&value, &identity, None, offset, "greedy"),
                    "final offset must be a complete aligned 20-pair chunk",
                );
            }
        }
    }

    #[test]
    fn declarations_validate_entire_ensemble_before_accepting_control() {
        let original = declaration(true);
        for field in [
            "candidate_radiant_runtime_sha256",
            "candidate_dire_runtime_sha256",
            "candidate_routing",
            "candidate_policy_sha256",
            "control_runtime_sha256",
        ] {
            for replacement in [None, Some(Value::Null), Some(json!("tampered"))] {
                let mut value = original.clone();
                value.as_object_mut().unwrap().remove(field);
                if let Some(replacement) = replacement {
                    value[field] = replacement;
                }
                rejects(
                    validate_declaration(&value, &"ef".repeat(32), None, 0, "greedy"),
                    field,
                );
            }
        }
        for field in [
            "candidate_radiant_runtime_sha256",
            "candidate_dire_runtime_sha256",
            "candidate_policy_sha256",
        ] {
            let mut value = original.clone();
            value[field] = json!("01".repeat(32));
            rejects(
                validate_declaration(&value, &"ef".repeat(32), None, 0, "greedy"),
                "candidate policy SHA256 does not bind ordered experts",
            );
        }
        let mut swapped = original.clone();
        swapped["candidate_radiant_runtime_sha256"] =
            original["candidate_dire_runtime_sha256"].clone();
        swapped["candidate_dire_runtime_sha256"] =
            original["candidate_radiant_runtime_sha256"].clone();
        rejects(
            validate_declaration(&swapped, &"ef".repeat(32), None, 0, "greedy"),
            "candidate policy SHA256 does not bind ordered experts",
        );
        swapped["candidate_runtime_sha256"] = json!("ab".repeat(32));
        rejects(
            validate_declaration(&swapped, &"ef".repeat(32), None, 0, "greedy"),
            "mixed single and ensemble candidate",
        );
    }

    #[test]
    fn declarations_reject_wrong_identity_experts_scope_and_legacy_corruption() {
        let value = declaration(true);
        let identity = value["candidate_policy_sha256"].as_str().unwrap();
        for (radiant, dire) in [("cd", "ab"), ("01", "cd"), ("ab", "01")] {
            rejects(
                validate_declaration(
                    &value,
                    identity,
                    Some((&radiant.repeat(32), &dire.repeat(32))),
                    0,
                    "greedy",
                ),
                "ensemble identity or experts differ from declaration",
            );
        }
        rejects(
            validate_declaration(
                &value,
                &"ef".repeat(32),
                Some((&"ab".repeat(32), &"cd".repeat(32))),
                0,
                "greedy",
            ),
            "ensemble identity or experts differ from declaration",
        );
        rejects(
            validate_declaration(&value, identity, None, 0, "greedy"),
            "single runtime is not preselected",
        );
        for (field, replacement) in [
            ("set", json!("autonomous-validation")),
            ("games_per_model", json!(201)),
            ("candidate_runtime_sha256", Value::Null),
            ("control_runtime_sha256", json!("bad")),
        ] {
            let mut value = declaration(false);
            value[field] = replacement;
            rejects(
                validate_declaration(&value, &"ab".repeat(32), None, 0, "greedy"),
                field,
            );
        }
    }

    #[test]
    fn declarations_reject_partial_ensemble_fields_and_experts_under_legacy_manifest() {
        for field in [
            "candidate_radiant_runtime_sha256",
            "candidate_dire_runtime_sha256",
            "candidate_routing",
            "candidate_policy_sha256",
        ] {
            let mut value = declaration(false);
            value[field] = Value::Null;
            rejects(
                validate_declaration(&value, &"ef".repeat(32), None, 0, "greedy"),
                "mixed single and ensemble candidate",
            );
            value
                .as_object_mut()
                .unwrap()
                .remove("candidate_runtime_sha256");
            rejects(
                validate_declaration(&value, &"ef".repeat(32), None, 0, "greedy"),
                "candidate_radiant_runtime_sha256",
            );
        }
        rejects(
            validate_declaration(
                &declaration(false),
                &"ab".repeat(32),
                Some((&"ab".repeat(32), &"cd".repeat(32))),
                0,
                "greedy",
            ),
            "ensemble identity or experts differ from declaration",
        );
        rejects(
            validate_declaration(&declaration(false), &"01".repeat(32), None, 0, "greedy"),
            "single runtime is not preselected",
        );
        rejects(
            validate_declaration(&declaration(false), "bad", None, 0, "greedy"),
            "evaluation identity must be 64 hex digits",
        );
    }

    #[test]
    fn blend_alpha_zero_copies_dire_and_one_copies_radiant_bit_exactly() {
        let radiant: Vec<_> = (0..MODEL_PARAMETER_COUNT)
            .map(|index| [0.0, -0.0, f32::MAX, f32::MIN_POSITIVE][index % 4])
            .collect();
        let dire: Vec<_> = radiant.iter().rev().copied().collect();
        for (alpha, expected) in [(0.0, &dire), (-0.0, &dire), (1.0, &radiant)] {
            let blended = blend_parameters(&radiant, &dire, alpha).unwrap();
            assert_eq!(blended.len(), MODEL_PARAMETER_COUNT);
            assert!(
                blended
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
            );
        }
    }

    #[test]
    fn blend_alpha_weights_radiant_and_f64_intermediates_avoid_extreme_value_overflow() {
        let mut radiant = vec![0.0; MODEL_PARAMETER_COUNT];
        let mut dire = vec![8.0; MODEL_PARAMETER_COUNT];
        radiant[0] = -f32::MAX;
        dire[0] = f32::MAX;
        for (alpha, expected) in [(0.25, 6.0), (0.5, 4.0), (0.75, 2.0)] {
            let blended = blend_parameters(&radiant, &dire, alpha).unwrap();
            assert!(blended[0].is_finite());
            assert_eq!(blended[0], f32::MAX * (1.0 - 2.0 * alpha));
            assert!(blended[1..].iter().all(|value| *value == expected));
        }
    }

    #[test]
    fn blend_rejects_bad_dimensions_alpha_and_nonfinite_values_even_at_endpoints() {
        let valid = vec![0.0; MODEL_PARAMETER_COUNT];
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
            rejects(
                blend_parameters(&valid, &valid, alpha),
                "blend alpha must be finite in 0..=1",
            );
        }
        for length in [0, MODEL_PARAMETER_COUNT - 1, MODEL_PARAMETER_COUNT + 1] {
            let invalid = vec![0.0; length];
            rejects(
                blend_parameters(&invalid, &valid, 0.5),
                "blend requires exact MODEL_PARAMETER_COUNT",
            );
            rejects(
                blend_parameters(&valid, &invalid, 0.5),
                "blend requires exact MODEL_PARAMETER_COUNT",
            );
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = valid.clone();
            invalid[MODEL_PARAMETER_COUNT - 1] = value;
            for alpha in [0.0, 0.5, 1.0] {
                rejects(
                    blend_parameters(&invalid, &valid, alpha),
                    "blend parameters must be finite",
                );
                rejects(
                    blend_parameters(&valid, &invalid, alpha),
                    "blend parameters must be finite",
                );
            }
        }
    }
}
