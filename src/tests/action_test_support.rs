use super::*;

/// The pre-optimization cell-centre landing scan, kept as the differential oracle.
#[cfg(test)]
fn nearest_landing_cell_reference(
    passability: &StaticPassability,
    center: Vec2,
    team: Team,
) -> Option<Vec2> {
    let (center_x, center_y) = passability.cell_of(center)?;
    let mut best: Option<(i64, usize, Vec2)> = None;
    let start_y = center_y.saturating_sub(LANDING_SEARCH_CELLS);
    let start_x = center_x.saturating_sub(LANDING_SEARCH_CELLS);
    let end_y = center_y
        .saturating_add(LANDING_SEARCH_CELLS)
        .min(passability.axis - 1);
    let end_x = center_x
        .saturating_add(LANDING_SEARCH_CELLS)
        .min(passability.axis - 1);
    for cell_y in start_y..=end_y {
        for cell_x in start_x..=end_x {
            let position = canonical_cell_position(passability, cell_x, cell_y, team);
            if !passability.walkable(position) {
                continue;
            }
            let distance = center.distance_squared(position);
            let cell_index = cell_y * passability.axis + cell_x;
            let cell_index = if team == Team::Dire {
                passability.axis * passability.axis - 1 - cell_index
            } else {
                cell_index
            };
            if best.is_none_or(|current| (distance, cell_index) < (current.0, current.1)) {
                best = Some((distance, cell_index, position));
            }
        }
    }
    best.map(|(_, _, position)| position)
}

/// Runs the direct-grid scan and the reference scan on one synthetic grid.
#[cfg(test)]
pub(crate) fn nearest_landing_cell_pair_for_test(
    axis: usize,
    open: Vec<bool>,
    center: Vec2,
    team: Team,
) -> (Option<Vec2>, Option<Vec2>) {
    assert!(axis > 0);
    assert_eq!(open.len(), axis * axis);
    let passability = StaticPassability { axis, open };
    (
        nearest_landing_cell_reference(&passability, center, team),
        nearest_landing_cell(&passability, center, team),
    )
}

#[cfg(test)]
pub(crate) fn tree_points_for_test(
    tracker: &StateTracker,
    mut points: Vec<PointCandidate>,
) -> Vec<PointCandidate> {
    assert!(points.len() <= MAX_POINT_CANDIDATES);
    let current = tracker.current().expect("tree test snapshot");
    let passability = reconstruct_static_passability(tracker, current).expect("tree passability");
    let center = tracker.own_hero().expect("tree test hero").pos;
    add_tree_points(tracker, current, &passability, center, &mut points).expect("tree points");
    assert!(points.len() <= MAX_POINT_CANDIDATES);
    points
}
