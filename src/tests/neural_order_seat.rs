use super::{
    neural_order_contract::{DiagnosticPolicy, install_policy, seated},
    support::RecordingWire,
};
use crate::{PolicyModel, Wire, play_neural_on};
use bota_proto::{EntityId, Order, ServerMsg};

// Only the policy schedule is synthetic; production Neural owns all order bookkeeping.
pub(super) fn replay(
    mut messages: Vec<ServerMsg>,
    side: usize,
    schedule: &[(u32, DiagnosticPolicy)],
) -> RecordingWire {
    struct ScheduledWire<'a> {
        wire: RecordingWire,
        model: &'a PolicyModel,
        schedule: &'a [(u32, DiagnosticPolicy)],
    }
    impl Wire for ScheduledWire<'_> {
        fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
            let message = self.wire.hear()?;
            if let Some(ServerMsg::Events { tick, .. }) = &message {
                for (scheduled, policy) in self.schedule {
                    if tick == scheduled {
                        install_policy(self.model, *policy);
                    }
                }
            }
            Ok(message)
        }
        fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
            self.wire.order(unit, order)
        }
        fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
            self.wire.acknowledge(tick)
        }
    }
    assert!(messages.len() <= 32);
    assert!(schedule.len() <= 8);
    let mut final_view = messages
        .iter()
        .rev()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view.clone()),
            _ => None,
        })
        .expect("final snapshot");
    final_view.tick += 1;
    let limit = final_view.tick;
    messages.push(ServerMsg::Snapshot { view: final_view });
    let model = PolicyModel::fresh(10_092_101).expect("diagnostic model");
    let mut wire = ScheduledWire {
        model: &model,
        schedule,
        wire: RecordingWire {
            messages: messages.into(),
            orders: Vec::new(),
            acknowledgements: Vec::new(),
        },
    };
    play_neural_on(&mut wire, seated(side), Some(limit), &model).expect("production Neural replay");
    assert!(wire.wire.messages.is_empty());
    wire.wire
}
