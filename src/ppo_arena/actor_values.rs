use super::*;

const _: () = assert!(MAX_ACTOR_ENVIRONMENTS <= 64);

/// One round of metadata only; frames and retained choices stay in their existing owners.
/// Delaying terminal entries too preserves the original rollout insertion order.
pub(super) struct PendingRound {
    entries: Vec<(usize, CompletedAdvance, &'static str)>,
    worlds: usize,
}

impl PendingRound {
    pub(super) fn new(worlds: usize) -> Self {
        assert!(worlds > 0);
        assert!(worlds <= MAX_ACTOR_ENVIRONMENTS);
        Self {
            entries: Vec::with_capacity(worlds),
            worlds,
        }
    }

    pub(super) fn push(
        &mut self,
        stream: usize,
        completed: CompletedAdvance,
        opponent: &'static str,
    ) {
        assert!(stream < self.worlds);
        assert!(self.entries.len() < self.worlds);
        assert!(self.entries.last().is_none_or(|entry| entry.0 < stream));
        self.entries.push((stream, completed, opponent));
    }

    pub(super) fn finish_sampled(
        &mut self,
        streams: &mut [EpisodeStream],
        stream_base: usize,
        samples: (&[usize], &[PpoPolicyChoice]),
        rollout: &mut PpoRollout,
        report: &mut PpoSmokeReport,
    ) -> Result<(), PpoError> {
        let (active, choices) = samples;
        assert_eq!(active.len(), choices.len());
        assert!(active.len() <= self.worlds);
        let mut values = [None; MAX_ACTOR_ENVIRONMENTS];
        for (&stream, choice) in active.iter().zip(choices) {
            assert!(stream < self.worlds);
            assert!(values[stream].is_none());
            values[stream] = Some(choice.value);
        }
        self.finish(streams, stream_base, rollout, report, |stream| {
            values[stream].ok_or(PpoError::InvalidTransition("pending actor bootstrap row"))
        })
    }

    pub(super) fn finish_fallback(
        &mut self,
        model: &PolicyModel,
        streams: &mut [EpisodeStream],
        stream_base: usize,
        prepared: &[Option<(FeatureFrame, ActionSpace)>],
        rollout: &mut PpoRollout,
        report: &mut PpoSmokeReport,
    ) -> Result<(), PpoError> {
        assert_eq!(prepared.len(), self.worlds);
        // Bounded windows can end before the next sample; never consume another actor draw.
        self.finish(streams, stream_base, rollout, report, |stream| {
            let (frame, _) = prepared[stream]
                .as_ref()
                .ok_or(PpoError::InvalidTransition("pending actor bootstrap frame"))?;
            let output = model
                .evaluate_batch(std::slice::from_ref(frame))
                .map_err(text_error)?;
            assert_eq!(output.len(), 1);
            Ok(output[0].value)
        })
    }

    fn finish(
        &mut self,
        streams: &mut [EpisodeStream],
        stream_base: usize,
        rollout: &mut PpoRollout,
        report: &mut PpoSmokeReport,
        mut next_value: impl FnMut(usize) -> Result<f32, PpoError>,
    ) -> Result<(), PpoError> {
        assert_eq!(streams.len(), self.worlds);
        assert!(self.entries.len() <= self.worlds);
        for (stream, completed, opponent) in self.entries.drain(..) {
            let state = &mut streams[stream];
            let value = if !state.done && state.should_flush() {
                Some(next_value(stream)?)
            } else {
                None
            };
            finish_advance_from_parts(
                state,
                stream_base + stream,
                completed,
                value,
                opponent,
                rollout,
                report,
            )?;
        }
        Ok(())
    }
}
