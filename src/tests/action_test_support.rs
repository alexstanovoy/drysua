use super::*;

pub(crate) fn landing_cell_for_test(open: [bool; 4], center: Vec2, team: Team) -> Option<Vec2> {
    let passability = StaticPassability {
        axis: 2,
        open: open.to_vec(),
    };
    nearest_landing_cell(&passability, center, team)
}

#[test]
fn landing_grid_ties_and_outside_or_blocked_boundaries() {
    for team in [Team::Radiant, Team::Dire] {
        let shift = i32::from(team == Team::Dire);
        let position = |coordinate| Vec2 {
            x: Fixed {
                raw: Fixed::from_int(coordinate).raw - shift,
            },
            y: Fixed {
                raw: Fixed::from_int(coordinate).raw - shift,
            },
        };
        for (open, center, expected) in [
            (
                [true; 4],
                position(64),
                Some(position(if team == Team::Radiant { 32 } else { 96 })),
            ),
            (
                [true, false, false, false],
                position(120),
                Some(position(32)),
            ),
            ([false; 4], position(64), None),
            ([true; 4], Vec2::from_ints(-1, 0), None),
            ([true; 4], Vec2::from_ints(128, 128), None),
        ] {
            assert_eq!(
                crate::action::landing_cell_for_test(open, center, team),
                expected
            );
        }
    }
}
