use super::*;

#[cfg(test)]
impl RaggedFeatureHeader {
    pub(crate) fn corrupt_unit_offset_for_test(&mut self) {
        self.units.offset = u32::MAX;
    }
}

impl RaggedFeatureArena {
    #[cfg(test)]
    pub(crate) fn stored_rows(&self) -> usize {
        self.units.len()
            + self.remembered_units.len()
            + self.points.len()
            + self.abilities.len()
            + self.items.len()
            + self.projectiles.len()
            + self.loot.len()
    }
}
