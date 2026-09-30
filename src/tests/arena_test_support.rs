use super::*;

impl Arena {
    #[cfg(test)]
    /// Read access to the native world for scenario assertions.
    pub(crate) fn world_for_test(&self) -> &World {
        &self.world
    }

    pub(crate) fn configure_for_test(&mut self, configure: impl FnOnce(&mut World)) -> ArenaStep {
        configure(&mut self.world);
        self.world.settle();
        self.world.lay_passability();
        let mut messages = vec![Vec::new(); self.picks.len()];
        self.append_tick_messages(&mut messages, &[]);
        assert_eq!(messages.len(), self.seat_count());
        ArenaStep { messages }
    }
}
