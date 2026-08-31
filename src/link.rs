use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use bota_proto::{
    ClientMsg, EntityId, FrameReader, Order, PlayerId, Role, ServerMsg, SlotId, TickMode,
    encode_frame_to_vec,
};

use crate::{SHADOW_FIEND, Wire};

const READ_BUFFER_LEN: usize = 64 * 1024;
const MAX_MESSAGES_BEFORE_WELCOME: usize = 64;
const MAX_NAME_LEN: usize = 64;
const MAX_RESOLVED_ADDRESSES: usize = 16;
const SERVER_IO_TIMEOUT: Duration = Duration::from_secs(300);

/// The seat terms returned by the server.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seated {
    /// The connection identity assigned by the server.
    pub player: PlayerId,
    /// The seat assigned to drysua.
    pub slot: SlotId,
    /// Simulation ticks per second.
    pub tick_rate: u16,
    /// The server tick advancement mode.
    pub mode: TickMode,
}

/// One framed TCP connection to a match server.
pub struct Link {
    stream: TcpStream,
    reader: FrameReader,
    next_sequence: u32,
}

impl Link {
    /// Connects, takes a bot seat, picks Shadow Fiend, and becomes ready.
    pub fn join(address: &str, name: &str) -> std::io::Result<(Self, Seated)> {
        Self::join_with_timeout(address, name, SERVER_IO_TIMEOUT)
    }

    /// Connects with one timeout for establishing and using the connection.
    pub fn join_with_timeout(
        address: &str,
        name: &str,
        timeout: Duration,
    ) -> std::io::Result<(Self, Seated)> {
        validate_name(name)?;
        if timeout.is_zero() {
            return Err(std::io::Error::other("server I/O timeout must be positive"));
        }
        let stream = connect(address, timeout)?;
        stream
            .set_nodelay(true)
            .map_err(|error| io_context("failed to enable TCP_NODELAY", error))?;
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|error| io_context("failed to set read timeout", error))?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(|error| io_context("failed to set write timeout", error))?;
        let mut link = Self {
            stream,
            reader: FrameReader::new(),
            next_sequence: 0,
        };
        link.send(&ClientMsg::Hello {
            role: Role::Bot,
            name: name.to_owned(),
        })?;
        let seated = link.wait_for_welcome()?;
        link.send(&ClientMsg::PickHero { hero: SHADOW_FIEND })?;
        link.send(&ClientMsg::SetReady(true))?;
        Ok((link, seated))
    }

    fn wait_for_welcome(&mut self) -> std::io::Result<Seated> {
        for _ in 0..MAX_MESSAGES_BEFORE_WELCOME {
            let message = self.hear()?.ok_or_else(|| {
                std::io::Error::other("server closed the connection before Welcome")
            })?;
            if let ServerMsg::Welcome {
                player_id,
                slot,
                tick_rate,
                mode,
            } = message
            {
                let slot = slot.ok_or_else(|| {
                    std::io::Error::other("server accepted drysua without assigning a seat")
                })?;
                return Ok(Seated {
                    player: player_id,
                    slot,
                    tick_rate,
                    mode,
                });
            }
        }
        Err(std::io::Error::other(format!(
            "server sent {MAX_MESSAGES_BEFORE_WELCOME} messages without Welcome"
        )))
    }

    fn send(&mut self, message: &ClientMsg) -> std::io::Result<()> {
        let frame = encode_frame_to_vec(message).map_err(|error| {
            std::io::Error::other(format!("failed to encode client message: {error}"))
        })?;
        self.stream
            .write_all(&frame)
            .map_err(|error| io_context("failed to send client message", error))
    }
}

fn connect(address: &str, timeout: Duration) -> std::io::Result<TcpStream> {
    let addresses = address.to_socket_addrs().map_err(|error| {
        std::io::Error::new(
            error.kind(),
            format!("failed to resolve server address {address}: {error}"),
        )
    })?;
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| std::io::Error::other("server I/O timeout is too large"))?;
    let mut last_error = None;
    for resolved in addresses.take(MAX_RESOLVED_ADDRESSES) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match TcpStream::connect_timeout(&resolved, remaining) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = Some(error),
        }
    }
    let detail = last_error.map_or_else(
        || "the address resolved to no usable socket".to_string(),
        |error| error.to_string(),
    );
    Err(std::io::Error::other(format!(
        "failed to connect to server at {address} within {timeout:?}: {detail}"
    )))
}

impl Wire for Link {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        loop {
            match self.reader.next_message::<ServerMsg>() {
                Ok(Some(message)) => return Ok(Some(message)),
                Ok(None) => {}
                Err(error) => {
                    return Err(std::io::Error::other(format!(
                        "failed to decode server message: {error}"
                    )));
                }
            }
            let mut buffer = [0_u8; READ_BUFFER_LEN];
            let read = self
                .stream
                .read(&mut buffer)
                .map_err(|error| io_context("failed to read server message", error))?;
            if read == 0 {
                return Ok(None);
            }
            self.reader.push(&buffer[..read]);
        }
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("order sequence number exhausted"))?;
        let sequence = self.next_sequence;
        self.send(&ClientMsg::Order {
            seq: sequence,
            unit,
            order,
        })?;
        Ok(sequence)
    }

    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        self.send(&ClientMsg::Ack { tick })
    }
}

fn validate_name(name: &str) -> std::io::Result<()> {
    if name.is_empty() {
        return Err(std::io::Error::other("bot name must not be empty"));
    }
    if name.len() > MAX_NAME_LEN {
        return Err(std::io::Error::other(format!(
            "bot name is {} bytes; maximum is {MAX_NAME_LEN}",
            name.len()
        )));
    }
    Ok(())
}

fn io_context(context: &str, error: std::io::Error) -> std::io::Error {
    std::io::Error::new(error.kind(), format!("{context}: {error}"))
}
