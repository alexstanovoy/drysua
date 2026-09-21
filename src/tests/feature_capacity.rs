use super::*;

#[test]
fn standard_feature_rows_reproduce_growth_past_logical_capacity() {
    let rows = [[1.0]; 3];
    let mut arena = Vec::new();

    let range = append_feature_rows(&mut arena, &rows, 0, 1).expect("three rows");

    assert_eq!(arena.len(), 3);
    assert_eq!(range.count, 3);
    assert!(
        arena.capacity() > 3,
        "ordinary Vec growth exceeds the row limit"
    );
}

#[test]
fn bounded_feature_rows_grow_geometrically_and_stop_at_exact_limit() {
    let rows = [[1.0], [0.0], [0.0]];
    let mut arena = Vec::new();
    for (index, capacity) in [1, 2, 4, 4, 8, 8, 8, 8, 9].into_iter().enumerate() {
        reserve_feature_rows(&mut arena, &rows, 0, 3).expect("bounded reservation");
        let range = append_feature_rows(&mut arena, &rows, 0, 3).expect("reserved row");

        assert_eq!(arena.capacity(), capacity);
        assert_eq!(arena.len(), index + 1);
        assert_eq!(range.offset as usize, index);
        assert_eq!(range.count, 1);
    }

    let error = reserve_feature_rows(&mut arena, &rows, 0, 3).expect_err("tenth row");

    assert_eq!(error, "ragged feature arena capacity exceeded");
    assert_eq!(arena.len(), 9);
    assert_eq!(arena.capacity(), 9);
}

#[test]
fn bounded_feature_constructor_accepts_maximum_without_allocating_and_rejects_outside_profile() {
    let arena = RaggedFeatureArena::new_bounded(crate::PPO_ANNEALED_MAX_SAMPLES)
        .expect("maximum sample capacity");

    assert_eq!(arena_capacities(&arena), [0; 7]);
    for capacity in [0, crate::PPO_ANNEALED_MAX_SAMPLES + 1, usize::MAX] {
        let error = RaggedFeatureArena::new_bounded(capacity)
            .err()
            .expect("invalid sample capacity");
        assert_eq!(
            error,
            "bounded ragged feature sample capacity is outside 1..=46520"
        );
    }
}

#[test]
fn bounded_feature_arena_caps_every_vector_and_roundtrips_dense_frame() {
    let frame = dense_frame();
    let mut arena = RaggedFeatureArena::new_bounded(1).expect("one frame");

    let header = arena.push(&frame).expect("full frame");

    assert_eq!(arena_capacities(&arena), token_counts());
    assert_eq!(arena.expand(&header).expect("roundtrip"), frame);
    assert_eq!(
        arena.push(&frame).expect_err("second full frame"),
        "ragged feature arena capacity exceeded"
    );
    assert_eq!(arena_capacities(&arena), token_counts());
    assert_eq!(arena.expand(&header).expect("unchanged frame"), frame);
}

#[test]
fn standard_feature_arena_keeps_ordinary_vec_growth() {
    let frame = dense_frame();
    let mut arena = RaggedFeatureArena::new(1);
    let mut expected = Vec::new();
    append_feature_rows(&mut expected, &frame.units, unit_feature::TOKEN_PRESENT, 1)
        .expect("ordinary unit rows");

    let header = arena.push(&frame).expect("standard frame");

    assert_eq!(arena.units.capacity(), expected.capacity());
    assert!(arena.units.capacity() > UNIT_FEATURE_TOKENS);
    assert_eq!(arena.expand(&header).expect("standard roundtrip"), frame);
}

#[test]
fn bounded_feature_arena_rejects_late_row_overflow_before_appending_any_rows() {
    let mut frame = FeatureFrame::new();
    for row in &mut frame.loot {
        row[loot_feature::TOKEN_PRESENT] = 1.0;
    }
    let mut arena = RaggedFeatureArena::new_bounded(1).expect("one frame");
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
fn bounded_feature_row_dimensions_reject_zero_oversize_and_capacity_overflow() {
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
}

#[cfg(target_pointer_width = "64")]
#[test]
fn bounded_feature_row_capacity_rejects_u32_offset_limit_plus_one() {
    assert_eq!(
        bounded_feature_row_capacity::<1>(u32::MAX as usize + 1),
        Err("ragged feature arena offset capacity exceeds u32")
    );
}

#[test]
fn bounded_feature_rows_reject_invalid_presence_index_before_allocation() {
    let mut arena = Vec::new();

    let error = reserve_feature_rows(&mut arena, &[[1.0]; 3], 1, 1)
        .expect_err("presence index outside each row");

    assert_eq!(error, "ragged feature presence index out of range");
    assert_eq!(arena.len(), 0);
    assert_eq!(arena.capacity(), 0);
}

#[test]
fn bounded_feature_rows_reject_preexisting_overallocated_vector() {
    let mut arena = Vec::with_capacity(4);

    let error = reserve_feature_rows(&mut arena, &[[0.0]; 3], 0, 1)
        .expect_err("four allocated rows exceed maximum three");

    assert_eq!(error, "ragged feature allocated capacity exceeds maximum");
    assert_eq!(arena.len(), 0);
    assert_eq!(arena.capacity(), 4);
}

#[test]
fn bounded_feature_reservation_reports_allocation_failure_without_allocating() {
    let mut arena: Vec<IndexedFeatureRow<1>> = Vec::new();

    // Requesting usize::MAX nonzero-sized rows fails Vec's byte-capacity check, not the allocator.
    let error = reserve_feature_capacity(&mut arena, usize::MAX, usize::MAX)
        .expect_err("unrepresentable allocation");

    assert_eq!(error, "ragged feature arena allocation failed");
    assert_eq!(arena.len(), 0);
    assert_eq!(arena.capacity(), 0);
}

#[test]
fn annealed_feature_peak_counts_all_row_capacities_and_one_largest_reallocation() {
    let counts = [96u64, 32, 48, 14, 85, 32, 16];
    let sizes = [340u64, 340, 132, 100, 116, 84, 68];
    let mut total = 0;
    let mut largest = 0;
    for (tokens, size) in counts.into_iter().zip(sizes) {
        let rows = 46_520 * tokens;
        assert!(rows <= u64::from(u32::MAX));
        let bytes = rows * size;
        total += bytes;
        largest = largest.max(bytes);
    }

    assert_eq!(total, 3_018_775_840);
    assert_eq!(largest, 1_518_412_800);
    assert_eq!(ANNEALED_FEATURE_ARENA_PEAK_BYTES, total + largest);
    assert_eq!(ANNEALED_FEATURE_ARENA_PEAK_BYTES, 4_537_188_640);
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
    for row in frame.units.iter_mut().chain(&mut frame.remembered_units) {
        row[unit_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in &mut frame.points {
        row[point_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in &mut frame.abilities {
        row[ability_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in &mut frame.items {
        row[item_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in &mut frame.projectiles {
        row[projectile_feature::TOKEN_PRESENT] = 1.0;
    }
    for row in &mut frame.loot {
        row[loot_feature::TOKEN_PRESENT] = 1.0;
    }
    frame
}
