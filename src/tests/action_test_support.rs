use super::*;

pub(crate) fn landing_cell_for_test(open: [bool; 4], center: Vec2, team: Team) -> Option<Vec2> {
    let passability = StaticPassability {
        axis: 2,
        open: open.to_vec(),
        felled: Vec::new(),
        landings: Vec::new(),
    };
    nearest_landing_cell(&passability, center, team)
}
