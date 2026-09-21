use super::emit_episode_logs;
use std::cell::RefCell;

#[test]
fn prometheus_keeps_episode_audit_event_but_suppresses_reward_dump() {
    let output = RefCell::new(Vec::new());
    emit_episode_logs(
        true,
        || output.borrow_mut().push("episode: audit"),
        || output.borrow_mut().push("event=map2_episode_reward"),
    );
    assert_eq!(*output.borrow(), ["episode: audit"]);
}

#[test]
fn legacy_mode_keeps_audit_event_before_reward_dump() {
    let output = RefCell::new(Vec::new());
    emit_episode_logs(
        false,
        || output.borrow_mut().push("episode: audit"),
        || output.borrow_mut().push("event=map2_episode_reward"),
    );
    assert_eq!(
        *output.borrow(),
        ["episode: audit", "event=map2_episode_reward"]
    );
}
