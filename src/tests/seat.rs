use std::collections::VecDeque;

use bota_proto::{
    EntityId, MapId, MatchInfo, Order, Pick, PlayerId, ServerMsg, SlotId, Team, TickMode, WorldView,
};

use crate::{SHADOW_FIEND, Seated, Wire, play_on};

struct MockWire {
    messages: VecDeque<ServerMsg>,
    acknowledgements: Vec<u32>,
}

impl Wire for MockWire {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        Ok(self.messages.pop_front())
    }

    fn order(&mut self, _unit: Option<EntityId>, _order: Order) -> std::io::Result<u32> {
        panic!("Continue policy must not send orders")
    }

    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        self.acknowledgements.push(tick);
        Ok(())
    }
}

#[test]
fn seat_loop_acknowledges_first_lockstep_snapshot_and_sends_no_order() {
    let mut wire = mock_wire();
    let outcome = play_on(&mut wire, seated(TickMode::Lockstep), Some(1)).expect("seat plays");

    assert_eq!(outcome.ticks, 1);
    assert_eq!(wire.acknowledgements, [1]);
}

#[test]
fn seat_loop_does_not_acknowledge_realtime_snapshot() {
    let mut wire = mock_wire_with_mode(TickMode::Realtime);
    let outcome = play_on(&mut wire, seated(TickMode::Realtime), Some(1)).expect("seat plays");

    assert_eq!(outcome.ticks, 1);
    assert!(wire.acknowledgements.is_empty());
}

#[test]
fn seat_loop_rejects_match_mode_that_differs_from_welcome() {
    let mut wire = mock_wire();

    let error = play_on(&mut wire, seated(TickMode::Realtime), Some(1))
        .expect_err("different match modes must fail");

    assert_eq!(
        error.to_string(),
        "MatchStart mode Lockstep differs from Welcome mode Realtime"
    );
}

#[test]
fn seat_loop_rejects_tick_rate_that_differs_from_welcome() {
    let mut wire = mock_wire();
    let ServerMsg::MatchStart { info } = &mut wire.messages[0] else {
        panic!("first fixture message must start the match");
    };
    info.tick_rate = 60;

    let error = play_on(&mut wire, seated(TickMode::Lockstep), Some(1))
        .expect_err("different tick rates must fail");

    assert_eq!(
        error.to_string(),
        "MatchStart tick rate 60 differs from Welcome tick rate 30"
    );
}

#[test]
fn seat_loop_rejects_snapshot_for_another_team() {
    let mut wire = mock_wire();
    let ServerMsg::Snapshot { view } = &mut wire.messages[1] else {
        panic!("second fixture message must be a snapshot");
    };
    view.viewer = Some(Team::Dire);

    let error = play_on(&mut wire, seated(TickMode::Lockstep), Some(1))
        .expect_err("another team's snapshot must fail");

    assert_eq!(
        error.to_string(),
        "Snapshot viewer Some(Dire) differs from assigned team Some(Radiant)"
    );
}

#[test]
fn seat_loop_rejects_non_shadow_fiend_match_pick() {
    let mut wire = mock_wire();
    let ServerMsg::MatchStart { info } = &mut wire.messages[0] else {
        panic!("first fixture message must start the match");
    };
    info.picks[0].hero = bota_proto::HeroId(1);

    let error =
        play_on(&mut wire, seated(TickMode::Lockstep), Some(1)).expect_err("wrong hero must fail");

    assert_eq!(
        error.to_string(),
        "assigned slot 0 picked HeroId(1), expected Shadow Fiend HeroId(2)"
    );
}

fn mock_wire() -> MockWire {
    mock_wire_with_mode(TickMode::Lockstep)
}

fn mock_wire_with_mode(mode: TickMode) -> MockWire {
    MockWire {
        messages: VecDeque::from([
            ServerMsg::MatchStart {
                info: match_info(mode),
            },
            ServerMsg::Snapshot {
                view: world_view(1),
            },
        ]),
        acknowledgements: Vec::new(),
    }
}

fn seated(mode: TickMode) -> Seated {
    Seated {
        player: PlayerId(1),
        slot: SlotId(0),
        tick_rate: 30,
        mode,
    }
}

fn match_info(mode: TickMode) -> MatchInfo {
    MatchInfo {
        match_id: 7,
        map: MapId(1),
        tick_rate: 30,
        pregame_ticks: 0,
        trees: Vec::new(),
        terrain_cells: 0,
        terrain_rle: Vec::new(),
        opaque_cells: Vec::new(),
        mode,
        picks: vec![Pick {
            slot: SlotId(0),
            team: Team::Radiant,
            hero: SHADOW_FIEND,
        }],
        shop: Vec::new(),
    }
}

fn world_view(tick: u32) -> WorldView {
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units: Vec::new(),
        projectiles: Vec::new(),
        players: Vec::new(),
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
}
