use super::*;

impl<W: Write> PerformanceOutput<W> {
    #[cfg(test)]
    pub(crate) const fn failed(&self) -> bool {
        self.failed
    }

    #[cfg(test)]
    pub(crate) fn into_inner(self) -> W {
        self.writer
    }
}
