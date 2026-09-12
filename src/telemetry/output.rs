use std::fmt;
use std::io::Write;

pub(crate) struct PerformanceOutput<W> {
    writer: W,
    failed: bool,
}

impl<W: Write> PerformanceOutput<W> {
    pub(crate) const fn new(writer: W) -> Self {
        Self {
            writer,
            failed: false,
        }
    }

    pub(crate) fn emit(&mut self, record: &impl fmt::Display) {
        if !self.failed && writeln!(self.writer, "{record}").is_err() {
            // A broken diagnostic sink must not change orders or interrupt a match.
            self.failed = true;
        }
    }

    pub(crate) fn finish(&mut self) {
        if !self.failed && self.writer.flush().is_err() {
            self.failed = true;
        }
    }

    #[cfg(test)]
    pub(crate) const fn failed(&self) -> bool {
        self.failed
    }

    #[cfg(test)]
    pub(crate) fn into_inner(self) -> W {
        self.writer
    }
}
