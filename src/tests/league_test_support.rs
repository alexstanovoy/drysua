use super::*;

impl PolicySnapshot {
    #[cfg(test)]
    pub(crate) fn from_parameters_for_test(
        parameters: Vec<f32>,
        generation: u64,
    ) -> Result<Self, LeagueError> {
        Self::from_parameters(parameters, generation)
    }
}

impl League {
    #[cfg(test)]
    pub(crate) fn insert_evaluated_for_test(
        &mut self,
        snapshot: PolicySnapshot,
        score: f64,
        profile: CrossPlayProfile,
    ) -> Result<(), LeagueError> {
        self.insert(snapshot, score, profile, false)
    }
}
