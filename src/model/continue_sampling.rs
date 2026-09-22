use super::*;

impl PolicyModel {
    /// Early callbacks mutate disposable worlds only; RNG publication is still transactional.
    pub(crate) fn sample_batch_continue(
        &self,
        frames: &[FeatureFrame],
        spaces: &[ActionSpace],
        random: &mut [PpoRng],
        mut submit: impl FnMut(usize, PpoPolicyChoice) -> Result<(), ModelError>,
        #[cfg(test)] fail_late: bool,
    ) -> Result<Vec<Option<PpoPolicyChoice>>, ModelError> {
        validate_policy_batch(frames, spaces)?;
        validate_sampling_rng_count(frames.len(), random.len())?;
        let mut staged = random.to_vec();
        let _guard = self.read_parameter_lock()?;
        let state = self.forward_frames(frames)?;
        let base = self.sampling_base_logits(&state)?;
        let mut random_rows = Some(staged.as_mut_slice());
        let rows = initialize_sampling_rows(&base, spaces, &mut random_rows)?;
        let policy = self.policy_identity_locked();
        let mut early = vec![None; rows.len()];
        for (index, row) in rows.iter().enumerate() {
            if row.prefix.kind() != ActionKind::Continue {
                continue;
            }
            let target = BehavioralTarget::from_action(
                &frames[index],
                &spaces[index],
                StructuredAction::Continue,
            )
            .map_err(|error| ModelError::Backend(error.to_string()))?;
            if spaces[index]
                .decode(StructuredAction::Continue)
                .map_err(|error| ModelError::Backend(error.to_string()))?
                .is_some()
            {
                return Err(ModelError::InvalidModelState(
                    "Continue overlap transport order",
                ));
            }
            let (log_probability, entropy) = row.observed.statistics(&target)?;
            early[index] = Some((log_probability.to_bits(), entropy.to_bits()));
            submit(
                index,
                PpoPolicyChoice {
                    frame: frames[index].clone(),
                    target,
                    action: StructuredAction::Continue,
                    policy,
                    log_probability,
                    entropy,
                    value: row.value,
                },
            )?;
        }
        // No compaction: even worlds already terminal on the CPU retain their original rows.
        let selections =
            self.finish_selection_stages_locked(state, base, rows, spaces, random_rows)?;
        #[cfg(test)]
        if fail_late {
            return Err(ModelError::Backend(
                "injected Continue overlap late decoder failure".to_owned(),
            ));
        }
        let choices = finish_overlap_choices(frames, spaces, selections, &early, policy)?;
        random.clone_from_slice(&staged);
        Ok(choices)
    }
}

fn finish_overlap_choices(
    frames: &[FeatureFrame],
    spaces: &[ActionSpace],
    selections: Vec<BatchSelection>,
    early: &[Option<(u32, u32)>],
    policy: PolicyIdentity,
) -> Result<Vec<Option<PpoPolicyChoice>>, ModelError> {
    assert_eq!(frames.len(), selections.len());
    assert_eq!(early.len(), selections.len());
    let mut choices = Vec::with_capacity(selections.len());
    for (index, selection) in selections.into_iter().enumerate() {
        let target =
            BehavioralTarget::from_action(&frames[index], &spaces[index], selection.action)
                .map_err(|error| ModelError::Backend(error.to_string()))?;
        let (log_probability, entropy) = selection.observed.statistics(&target)?;
        if let Some(statistics) = early[index] {
            assert_eq!(selection.action, StructuredAction::Continue);
            assert_eq!(statistics, (log_probability.to_bits(), entropy.to_bits()));
            choices.push(None);
        } else {
            choices.push(Some(PpoPolicyChoice {
                frame: frames[index].clone(),
                target,
                action: selection.action,
                policy,
                log_probability,
                entropy,
                value: selection.value,
            }));
        }
    }
    Ok(choices)
}
