use std::error::Error;

use bota_proto::{MapId, ServerMsg, SlotId, Team};

use crate::persistence::training::PolicyOrderBookkeeping;
use crate::{
    ActionSpace, ActiveOrderUpdate, ActivePolicyOrder, FeatureEncoder, FeatureFrame, ItemReadiness,
    LocalPolicyState, OrderPersistence, PolicyModel, Request, StateTracker, StructuredAction,
    active_order_update_for_sent,
};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const ACTOR_DECISIONS: u32 = crate::MAP2_ACTOR_DECISIONS as u32;

struct NeuralSeat {
    tracker: StateTracker,
    encoder: FeatureEncoder,
    local: LocalPolicyState,
    persistence: OrderPersistence,
    order_bookkeeping: PolicyOrderBookkeeping,
    readiness: ItemReadiness,
    pending: Option<(u32, Option<ActivePolicyOrder>)>,
    decisions: u32,
    sequence: u32,
    fountain: Option<bota_proto::Vec2>,
    left_fountain_area: bool,
}

pub(crate) struct NeuralSeatOrderContractProbe(NeuralSeat);

impl NeuralSeatOrderContractProbe {
    /// Replays archived Map0/Map1 order fixtures without relabeling their metadata.
    pub(crate) fn new_historical(side: usize, messages: &[ServerMsg]) -> Self {
        assert!(side < 2);
        let Some(ServerMsg::MatchStart { info }) = messages.first() else {
            panic!("historical order-contract MatchStart");
        };
        assert!(matches!(info.map, MapId(0) | MapId(1)));
        let tracker = StateTracker::new(SlotId(side as u8), info).expect("historical tracker");
        Self(NeuralSeat::from_tracker(tracker, messages).expect("historical seat"))
    }

    pub(crate) fn observe(&mut self, messages: &[ServerMsg]) {
        assert!(
            self.0
                .observe(messages)
                .expect("NeuralSeat probe observations")
                .is_none()
        );
    }

    pub(crate) fn decide(
        &mut self,
        model: &PolicyModel,
        candidate: bool,
    ) -> (FeatureFrame, Option<Request>, Option<ActivePolicyOrder>) {
        let (action, space) = if candidate {
            self.0
                .order_bookkeeping
                .enable_candidate(&self.0.persistence)
                .expect("candidate role");
            self.0
                .neural_choice(model)
                .expect("NeuralSeat probe choice")
        } else {
            let space =
                ActionSpace::from_tracker_with_readiness(&self.0.tracker, &self.0.readiness)
                    .expect("legacy probe space");
            let frame = self.0.frame(&space).expect("legacy probe input");
            (
                model
                    .choose(&frame, &space)
                    .expect("legacy probe choice")
                    .action,
                space,
            )
        };
        let frame = self.0.frame(&space).expect("NeuralSeat probe input");
        let request = self
            .0
            .issue(action, &space)
            .expect("NeuralSeat probe send")
            .map(|issued| Request {
                seq: self.0.sequence,
                unit: issued.unit,
                order: issued.order,
            });
        (frame, request, self.0.local.active_order())
    }

    pub(crate) fn legacy_persistence(&self) -> OrderPersistence {
        self.0.persistence
    }
}

impl NeuralSeat {
    fn from_tracker(tracker: StateTracker, messages: &[ServerMsg]) -> Result<Self> {
        assert!(messages.len() <= 4);
        assert!(!messages.is_empty());
        let mut seat = Self {
            encoder: FeatureEncoder::new(&tracker),
            tracker,
            local: LocalPolicyState::new(0),
            persistence: OrderPersistence::default(),
            order_bookkeeping: PolicyOrderBookkeeping::Legacy,
            readiness: ItemReadiness::new(),
            pending: None,
            decisions: 0,
            sequence: 0,
            fountain: None,
            left_fountain_area: false,
        };
        seat.observe(messages)?;
        Ok(seat)
    }

    fn observe(&mut self, messages: &[ServerMsg]) -> Result<Option<Team>> {
        assert!(messages.len() <= 4);
        assert!(!messages.is_empty());
        let mut winner = None;
        for message in messages {
            match message {
                ServerMsg::Snapshot { view } => {
                    let previous = self.tracker.own_hero().map(|hero| hero.id);
                    self.tracker.observe_snapshot(view)?;
                    self.observe_opening(view);
                    if previous != self.tracker.own_hero().map(|hero| hero.id) {
                        self.persistence.clear_body_for(None);
                        self.order_bookkeeping.clear_body_for(None);
                        self.local.set_active_order(view.tick, None)?;
                        self.pending = None;
                    }
                }
                ServerMsg::Events { tick, events } => {
                    self.tracker.observe_events(*tick, events)?;
                    self.order_bookkeeping.reconcile(
                        &self.tracker,
                        &mut self.local,
                        &mut self.pending,
                    )?;
                    self.encoder.observe(&self.tracker)?;
                }
                ServerMsg::OrderRejected { seq, .. } => {
                    self.persistence.observe_rejection(*seq);
                    self.order_bookkeeping.observe_rejection(*seq);
                    self.readiness.note_rejected(*seq);
                    if let Some((pending, previous)) = self.pending
                        && pending == *seq
                    {
                        self.local.restore_active_order(
                            self.tracker.current().ok_or("snapshot missing")?.tick,
                            previous,
                        )?;
                        self.pending = None;
                    }
                }
                ServerMsg::MatchOver {
                    winner: team,
                    stats,
                } => {
                    assert_eq!(stats.slots.len(), 2);
                    assert_eq!(
                        stats.duration,
                        self.tracker.current().ok_or("snapshot missing")?.tick
                    );
                    winner = Some(*team);
                }
                ServerMsg::MatchStart { .. } => {}
                _ => return Err("unexpected builtin seat message".into()),
            }
        }
        Ok(winner)
    }

    fn frame(&mut self, space: &ActionSpace) -> Result<FeatureFrame> {
        let mut frame = FeatureFrame::new();
        self.encoder.encode(
            &self.tracker,
            space,
            &self.readiness,
            &self.local,
            &mut frame,
        )?;
        assert!(frame.matches_action_space(space));
        assert!(frame.is_finite());
        Ok(frame)
    }

    fn observe_opening(&mut self, view: &bota_proto::WorldView) {
        if self.fountain.is_none() {
            self.fountain = view
                .units
                .iter()
                .find(|unit| {
                    unit.team == self.tracker.team() && unit.kind == bota_proto::UnitKind::Fountain
                })
                .map(|unit| unit.pos);
        }
        if view.tick <= 3000
            && let Some(origin) = self.fountain
            && let Some(hero) = self.tracker.own_hero()
        {
            self.left_fountain_area |= !hero.pos.within(origin, bota_proto::Fixed::from_int(1200));
        }
    }

    /// Predicts without opting a Teacher transport into candidate deduplication.
    fn neural_choice(&mut self, model: &PolicyModel) -> Result<(StructuredAction, ActionSpace)> {
        self.order_bookkeeping.enable_observer(&self.persistence)?;
        self.order_bookkeeping
            .reconcile(&self.tracker, &mut self.local, &mut self.pending)?;
        let space = ActionSpace::from_tracker_with_readiness(&self.tracker, &self.readiness)?;
        let frame = self.frame(&space)?;
        let action = model.choose(&frame, &space)?.action;
        assert!(space.allows(action));
        Ok((action, space))
    }

    fn issue(
        &mut self,
        action: StructuredAction,
        space: &ActionSpace,
    ) -> Result<Option<crate::IssuedOrder>> {
        assert!(self.decisions < ACTOR_DECISIONS);
        self.decisions += 1;
        self.local.note_decision(space.tick(), action.kind())?;
        let persistence = self.order_bookkeeping.transport(&self.persistence);
        let Some(issued) = persistence.should_send(space.decode(action)?) else {
            return Ok(None);
        };
        let previous = self.local.active_order();
        self.sequence += 1;
        let preserves = self.order_bookkeeping.record_sent(
            &mut self.persistence,
            self.sequence,
            issued,
            &self.tracker,
        )?;
        self.readiness.note_sent(self.sequence, issued, space);
        let update = if preserves {
            ActiveOrderUpdate::Preserve
        } else {
            active_order_update_for_sent(
                self.order_bookkeeping.effective(&self.persistence),
                issued.unit,
                self.sequence,
                action.kind(),
            )
        };
        match update {
            ActiveOrderUpdate::Preserve => {}
            ActiveOrderUpdate::Replace(None) if previous.is_none() => self.pending = None,
            ActiveOrderUpdate::Replace(next) => {
                if let Some(kind) = next {
                    self.local
                        .set_active_order_from_issued(space.tick(), kind, issued)?;
                } else {
                    self.local.set_active_order(space.tick(), None)?;
                }
                self.pending = Some((self.sequence, previous));
            }
        }
        Ok(Some(issued))
    }
}
