use bota_proto::{EntityId, Order, ServerMsg};

/// A framed match connection used by the seat loop.
pub trait Wire {
    /// Reads the next server message, or no message after a clean close.
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>>;

    /// Sends one order and returns its connection sequence number.
    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32>;

    /// Acknowledges one snapshot tick.
    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()>;
}
