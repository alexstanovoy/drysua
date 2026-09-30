//! Encoder inputs packed once per frame, on the thread that encoded it.
//!
//! A row holds one frame's conditioned encoder parts in device buffer order.
//! Batches concatenate rows part by part, so a batch built from rows equals the
//! historical per-batch staging value for value.

use super::*;

/// Per-frame lengths of the encoder input parts, in device buffer order.
pub(super) const PART_LENGTHS: [usize; ENCODER_PARTS] = [
    ENCODER_UNIT_TOKENS * UNIT_FEATURES,
    ENCODER_UNIT_TOKENS,
    ENCODER_UNIT_TOKENS,
    ENCODER_UNIT_TOKENS,
    ENCODER_UNIT_TOKENS,
    ENCODER_UNIT_TOKENS,
    ENCODER_UNIT_TOKENS,
    OWN_UNIT_FEATURE_TOKENS * UNIT_FEATURES,
    OWN_UNIT_FEATURE_TOKENS,
    ABILITY_FEATURE_TOKENS * ABILITY_FEATURES,
    ABILITY_FEATURE_TOKENS,
    ITEM_FEATURE_TOKENS * ITEM_FEATURES,
    ITEM_FEATURE_TOKENS,
    POINT_FEATURE_TOKENS * POINT_FEATURES,
    POINT_FEATURE_TOKENS,
    PROJECTILE_FEATURE_TOKENS * PROJECTILE_FEATURES,
    PROJECTILE_FEATURE_TOKENS,
    LOOT_FEATURE_TOKENS * LOOT_FEATURES,
    LOOT_FEATURE_TOKENS,
    ENCODER_SCALARS,
];
pub(super) const ENCODER_PARTS: usize = 3 + UNIT_GROUPS + 12;

/// Scalars of one packed row.
pub const ENCODER_ROW_ELEMENTS: usize = {
    let mut total = 0;
    let mut index = 0;
    while index < ENCODER_PARTS {
        total += PART_LENGTHS[index];
        index += 1;
    }
    total
};
const _: () = assert!(ENCODER_ROW_ELEMENTS < 32 * 1024);

pub(super) const fn part_offset(part: usize) -> usize {
    let mut total = 0;
    let mut index = 0;
    while index < part {
        total += PART_LENGTHS[index];
        index += 1;
    }
    total
}

/// One frame's conditioned encoder input and actor side.
#[derive(Clone)]
pub struct EncoderRow {
    values: Box<[f32]>,
    radiant: bool,
}

impl Default for EncoderRow {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderRow {
    /// A zeroed dire row; one allocation reused by every later [`EncoderRow::pack`].
    pub fn new() -> Self {
        Self {
            values: vec![0.0; ENCODER_ROW_ELEMENTS].into_boxed_slice(),
            radiant: false,
        }
    }

    /// A packed copy of one frame.
    pub fn from_frame(frame: &FeatureFrame) -> Result<Self, ModelError> {
        let mut row = Self::new();
        row.pack(frame)?;
        Ok(row)
    }

    /// Overwrites this row with one finite frame whose side one-hot is valid.
    pub fn pack(&mut self, frame: &FeatureFrame) -> Result<(), ModelError> {
        if !frame.is_finite() {
            return Err(ModelError::NonFiniteFrame { index: 0 });
        }
        self.radiant = side_actors::side_row(frame, 0)? == 1;
        self.pack_units(frame);
        let own = self.parts(7);
        pack_present_rows(
            &frame.own_units,
            unit_feature::TOKEN_PRESENT,
            &[(unit_feature::KIND_TOKEN, 12.0)],
            None,
            own,
        );
        self.pack_tokens(frame);
        pack_scalars(frame, self.part(19));
        Ok(())
    }

    pub(crate) const fn radiant(&self) -> bool {
        self.radiant
    }

    pub(super) fn part_values(&self, part: usize) -> &[f32] {
        let offset = part_offset(part);
        &self.values[offset..offset + PART_LENGTHS[part]]
    }

    fn part(&mut self, part: usize) -> &mut [f32] {
        let offset = part_offset(part);
        &mut self.values[offset..offset + PART_LENGTHS[part]]
    }

    /// Two adjacent parts: token values then their presence mask.
    fn parts(&mut self, first: usize) -> (&mut [f32], &mut [f32]) {
        let offset = part_offset(first);
        let (values, mask) = self.values
            [offset..offset + PART_LENGTHS[first] + PART_LENGTHS[first + 1]]
            .split_at_mut(PART_LENGTHS[first]);
        (values, mask)
    }

    fn pack_units(&mut self, frame: &FeatureFrame) {
        let tokens = ENCODER_UNIT_TOKENS;
        let (values, masks) = self.values[..tokens * (UNIT_FEATURES + 1 + UNIT_GROUPS)]
            .split_at_mut(tokens * UNIT_FEATURES);
        let (presence, groups) = masks.split_at_mut(tokens);
        let rows = frame.units.iter().chain(frame.remembered_units.iter());
        for (token, row) in rows.enumerate() {
            let present = row[unit_feature::TOKEN_PRESENT] == 1.0;
            let target = &mut values[token * UNIT_FEATURES..(token + 1) * UNIT_FEATURES];
            if present {
                target.copy_from_slice(row);
            } else {
                target.fill(0.0);
            }
            let group = unit_group(row[unit_feature::KIND_TOKEN]);
            let mut any = false;
            for index in 0..UNIT_GROUPS {
                let member = present && group == Some(index);
                groups[index * tokens + token] = member as u8 as f32;
                any |= member;
            }
            presence[token] = any as u8 as f32;
        }
        condition_rows(
            values,
            UNIT_FEATURES,
            &[(unit_feature::KIND_TOKEN, 12.0)],
            None,
        );
    }

    fn pack_tokens(&mut self, frame: &FeatureFrame) {
        pack_present_rows(
            &frame.abilities,
            ability_feature::TOKEN_PRESENT,
            &[
                (ability_feature::BODY_TOKEN, 2.0),
                (ability_feature::SEMANTIC_SLOT_TOKEN, 8.0),
                (ability_feature::AIM_TOKEN, 5.0),
            ],
            Some((ability_feature::ID_TOKEN, 65_547.0)),
            self.parts(9),
        );
        pack_present_rows(
            &frame.items,
            item_feature::TOKEN_PRESENT,
            &[
                (item_feature::LOCATION_TOKEN, 5.0),
                (item_feature::SLOT_TOKEN, 64.0),
                (item_feature::AIM_TOKEN, 5.0),
                (item_feature::ATTRIBUTE_TOKEN, 3.0),
            ],
            Some((item_feature::ITEM_TOKEN, 65_536.0)),
            self.parts(11),
        );
        pack_present_rows(
            &frame.points,
            point_feature::TOKEN_PRESENT,
            &[
                (point_feature::SOURCE_TOKEN, 8.0),
                (point_feature::SOURCE_DIRECTION_TOKEN, 8.0),
                (point_feature::SOURCE_KIND_TOKEN, 12.0),
            ],
            None,
            self.parts(13),
        );
        pack_present_rows(
            &frame.projectiles,
            projectile_feature::TOKEN_PRESENT,
            &[],
            Some((projectile_feature::ABILITY_TOKEN, 65_547.0)),
            self.parts(15),
        );
        pack_present_rows(
            &frame.loot,
            loot_feature::TOKEN_PRESENT,
            &[],
            Some((loot_feature::ITEM_TOKEN, 65_536.0)),
            self.parts(17),
        );
    }
}

fn pack_present_rows<const TOKENS: usize, const FEATURES: usize>(
    rows: &[[f32; FEATURES]; TOKENS],
    presence: usize,
    categories: &[(usize, f32)],
    semantic_id: Option<(usize, f32)>,
    (values, mask): (&mut [f32], &mut [f32]),
) {
    assert_eq!(values.len(), TOKENS * FEATURES);
    assert_eq!(mask.len(), TOKENS);
    for (token, row) in rows.iter().enumerate() {
        let present = row[presence] == 1.0;
        let target = &mut values[token * FEATURES..(token + 1) * FEATURES];
        if present {
            target.copy_from_slice(row);
        } else {
            target.fill(0.0);
        }
        mask[token] = present as u8 as f32;
    }
    condition_rows(values, FEATURES, categories, semantic_id);
}

fn pack_scalars(frame: &FeatureFrame, values: &mut [f32]) {
    assert_eq!(values.len(), ENCODER_SCALARS);
    let (global, rest) = values.split_at_mut(GLOBAL_FEATURES);
    global.copy_from_slice(&frame.global);
    condition_rows(
        global,
        GLOBAL_FEATURES,
        &[
            (crate::global_feature::ROLE_TOKEN, 5.0),
            (crate::global_feature::LANE_TOKEN, 3.0),
            (crate::global_feature::ACTIVE_ORDER_KIND, 16.0),
            (crate::global_feature::ACTIVE_TARGET_KIND_TOKEN, 12.0),
        ],
        None,
    );
    let (history, rest) = rest.split_at_mut(HISTORY_SAMPLES * HISTORY_FEATURES);
    history.copy_from_slice(frame.history.as_flattened());
    let (policy, map) = rest.split_at_mut(MAX_POLICY_HISTORY * POLICY_HISTORY_FEATURES);
    policy.copy_from_slice(frame.policy_history.as_flattened());
    condition_rows(policy, POLICY_HISTORY_FEATURES, &[(3, 16.0)], None);
    map.copy_from_slice(&frame.map);
}

/// Concatenates rows part by part into one host buffer and its part lengths.
pub(super) fn assemble(rows: &[&EncoderRow]) -> (Vec<f32>, Vec<usize>) {
    assert!((1..=MODEL_PPO_MAX_MICROBATCH).contains(&rows.len()));
    let mut buffer = Vec::with_capacity(rows.len() * ENCODER_ROW_ELEMENTS);
    let mut lengths = Vec::with_capacity(ENCODER_PARTS);
    for (part, &length) in PART_LENGTHS.iter().enumerate() {
        let offset = part_offset(part);
        for row in rows {
            buffer.extend_from_slice(&row.values[offset..offset + length]);
        }
        lengths.push(length * rows.len());
    }
    assert_eq!(buffer.len(), rows.len() * ENCODER_ROW_ELEMENTS);
    (buffer, lengths)
}
