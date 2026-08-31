use std::time::Duration;

#[test]
fn link_rejects_zero_io_timeout() {
    let error = crate::Link::join_with_timeout("127.0.0.1:4455", "drysua", Duration::ZERO)
        .err()
        .expect("zero timeout must fail before connecting");

    assert_eq!(error.to_string(), "server I/O timeout must be positive");
}
