use super::*;

impl LogPublisher {
    #[cfg(test)]
    pub(crate) fn writer(&self) -> AsyncLogWriter {
        AsyncLogWriter {
            publisher: Some(self.clone()),
            line: LogLine::default(),
        }
    }
}
