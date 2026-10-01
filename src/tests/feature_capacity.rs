use super::*;

#[test]
fn ragged_arena_fills_exact_capacity_then_rejects_overflow_and_keeps_retained_frames() {
    let mut sparse = FeatureFrame::new();
    sparse.units[UNIT_FEATURE_TOKENS - 1][unit_feature::TOKEN_PRESENT] = 1.0;
    let mut unit_rows = [0; 7];
    unit_rows[0] = UNIT_FEATURE_TOKENS;
    for (case, frame, pushes, capacities) in [
        (
            "one unit row per frame",
            sparse,
            UNIT_FEATURE_TOKENS,
            unit_rows,
        ),
        ("dense frame", dense_frame(), 1, token_counts()),
    ] {
        let mut arena = RaggedFeatureArena::new(1).expect(case);
        let headers: Vec<_> = (0..pushes)
            .map(|_| arena.push(&frame).expect(case))
            .collect();
        assert_eq!(arena_capacities(&arena), capacities, "{case}");
        assert_eq!(
            arena.push(&frame).expect_err(case),
            "ragged feature arena capacity exceeded",
            "{case}"
        );
        assert_eq!(arena_capacities(&arena), capacities, "{case}");
        for header in &headers {
            assert_eq!(arena.expand(header).expect(case), frame, "{case}");
        }
    }
}

#[test]
fn ragged_arena_rejects_late_row_overflow_before_appending_any_rows() {
    let mut frame = FeatureFrame::new();
    for row in frame.loot.iter_mut() {
        row[loot_feature::TOKEN_PRESENT] = 1.0;
    }
    let mut arena = RaggedFeatureArena::new(1).expect("one frame");
    let header = arena.push(&frame).expect("loot rows");
    let mut overflow = frame.clone();
    overflow.units[0][unit_feature::TOKEN_PRESENT] = 1.0;

    let error = arena
        .push(&overflow)
        .expect_err("loot overflow after unit reservation");

    assert_eq!(error, "ragged feature arena capacity exceeded");
    assert!(arena.units.is_empty());
    assert_eq!(arena.loot.len(), LOOT_FEATURE_TOKENS);
    assert_eq!(arena.expand(&header).expect("unchanged loot frame"), frame);
}

#[test]
fn ragged_arena_rejects_a_corrupt_row_offset() {
    let mut arena = RaggedFeatureArena::new(1).expect("arena");
    let mut malformed = arena.push(&dense_frame()).expect("frame");
    malformed.corrupt_unit_offset_for_test();
    assert_eq!(
        arena.expand(&malformed).expect_err("invalid offset"),
        "ragged feature range is invalid"
    );
}

#[test]
fn ragged_arena_constructor_accepts_maximum_without_allocating_and_rejects_outside_range() {
    let arena = RaggedFeatureArena::new(crate::PPO_MAX_SAMPLES).expect("maximum sample capacity");

    assert_eq!(arena_capacities(&arena), [0; 7]);
    for capacity in [0, crate::PPO_MAX_SAMPLES + 1, usize::MAX] {
        let error = RaggedFeatureArena::new(capacity)
            .err()
            .expect("invalid sample capacity");
        assert_eq!(
            error, "ragged feature sample capacity is outside 1..=33280",
            "{capacity}"
        );
    }
}

#[test]
fn feature_row_capacity_rejects_zero_oversize_multiplication_and_u32_offset_overflow() {
    assert_eq!(
        bounded_feature_row_capacity::<0>(1),
        Err("ragged feature token count is outside 1..=65535")
    );
    assert_eq!(
        bounded_feature_row_capacity::<65_536>(1),
        Err("ragged feature token count is outside 1..=65535")
    );
    assert_eq!(
        bounded_feature_row_capacity::<2>(usize::MAX),
        Err("ragged feature arena capacity overflow")
    );
    assert_eq!(
        bounded_feature_row_capacity::<1>(u32::MAX as usize),
        Ok(u32::MAX as usize)
    );
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        bounded_feature_row_capacity::<1>(u32::MAX as usize + 1),
        Err("ragged feature arena offset capacity exceeds u32")
    );
}

#[test]
fn feature_row_reservation_failures_leave_the_vector_unallocated_or_unchanged() {
    type Reserve = fn(&mut Vec<IndexedFeatureRow<1>>) -> Result<(), &'static str>;
    let cases: [(&str, usize, Reserve, &str); 3] = [
        (
            "presence index outside each row",
            0,
            |rows| reserve_feature_rows(rows, &[[1.0]; 3], 1, 1),
            "ragged feature presence index out of range",
        ),
        (
            "four allocated rows exceed maximum three",
            4,
            |rows| reserve_feature_rows(rows, &[[0.0]; 3], 0, 1),
            "ragged feature allocated capacity exceeds maximum",
        ),
        (
            // usize::MAX nonzero-sized rows fail Vec's byte-capacity check, not the allocator.
            "unrepresentable allocation",
            0,
            |rows| reserve_feature_capacity(rows, usize::MAX, usize::MAX),
            "ragged feature arena allocation failed",
        ),
    ];
    for (case, allocated, reserve, expected) in cases {
        let mut rows = Vec::with_capacity(allocated);
        assert_eq!(reserve(&mut rows), Err(expected), "{case}");
        assert_eq!(rows.len(), 0, "{case}");
        assert_eq!(rows.capacity(), allocated, "{case}");
    }
}

fn arena_capacities(arena: &RaggedFeatureArena) -> [usize; 7] {
    [
        arena.units.capacity(),
        arena.remembered_units.capacity(),
        arena.points.capacity(),
        arena.abilities.capacity(),
        arena.items.capacity(),
        arena.projectiles.capacity(),
        arena.loot.capacity(),
    ]
}

fn token_counts() -> [usize; 7] {
    [
        UNIT_FEATURE_TOKENS,
        REMEMBERED_UNIT_FEATURE_TOKENS,
        POINT_FEATURE_TOKENS,
        ABILITY_FEATURE_TOKENS,
        ITEM_FEATURE_TOKENS,
        PROJECTILE_FEATURE_TOKENS,
        LOOT_FEATURE_TOKENS,
    ]
}

fn dense_frame() -> FeatureFrame {
    let mut frame = FeatureFrame::new();
    for row in frame
        .units
        .iter_mut()
        .chain(frame.remembered_units.iter_mut())
    {
        row[unit_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in frame.points.iter_mut() {
        row[point_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in frame.abilities.iter_mut() {
        row[ability_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in frame.items.iter_mut() {
        row[item_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in frame.projectiles.iter_mut() {
        row[projectile_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in frame.loot.iter_mut() {
        row[loot_feature::TOKEN_PRESENT] = 1.0;
    }
    frame
}
