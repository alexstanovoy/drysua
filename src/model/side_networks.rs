//! Whether Radiant and Dire share one network or own two complete ones, and the
//! routing of rows to networks. Observations stay team-canonical either way; a
//! row's observed side picks its actor heads (shared) or its network (separate).
use super::*;

/// How the two seat sides map onto networks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SideNetworks {
    /// One network: encoders, trunk, critic and embeddings are shared and only
    /// the actor heads are per side (model M25).
    #[default]
    Shared,
    /// Two complete networks, Radiant then Dire: a row is sampled, valued and
    /// trained only by its own side's network.
    Separate,
}

/// F32 parameters of one complete network without the shared layout's Dire heads.
pub(crate) const NETWORK_PARAMETER_COUNT: usize = 1_891_700;
/// Named tensors of one complete network.
const NETWORK_TENSORS: usize = 64;
const _: () =
    assert!(MODEL_PARAMETER_COUNT == NETWORK_PARAMETER_COUNT + side_actors::DIRE_HEAD_PARAMETERS);
const _: () = assert!(MODEL_PARAMETER_TENSORS == NETWORK_TENSORS + 24);

impl SideNetworks {
    /// The run-scope and runtime-metadata spelling.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Separate => "separate",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        [Self::Shared, Self::Separate]
            .into_iter()
            .find(|layout| layout.label() == label)
    }

    /// Exact F32 parameters of the layout.
    pub const fn parameter_count(self) -> usize {
        match self {
            Self::Shared => MODEL_PARAMETER_COUNT,
            Self::Separate => 2 * NETWORK_PARAMETER_COUNT,
        }
    }

    /// The layout whose parameter vectors have `count` elements; the two counts differ.
    pub(crate) fn from_parameter_count(count: usize) -> Option<Self> {
        [Self::Shared, Self::Separate]
            .into_iter()
            .find(|layout| layout.parameter_count() == count)
    }

    pub(super) const fn parameter_tensors(self) -> usize {
        match self {
            Self::Shared => MODEL_PARAMETER_TENSORS,
            Self::Separate => 2 * NETWORK_TENSORS,
        }
    }
}

macro_rules! dire_network_names {
    ($($name:literal),* $(,)?) => {
        /// Each tensor name of one network and its name in the separate Dire network.
        const DIRE_NETWORK_NAMES: [(&str, &str); NETWORK_TENSORS] =
            [$(($name, concat!("dire.", $name))),*];
    };
}

dire_network_names![
    "unit.0.weight",
    "unit.0.bias",
    "unit.1.weight",
    "unit.1.bias",
    "unit.2.weight",
    "unit.2.bias",
    "ability.0.weight",
    "ability.0.bias",
    "ability.1.weight",
    "ability.1.bias",
    "item.0.weight",
    "item.0.bias",
    "item.1.weight",
    "item.1.bias",
    "point.0.weight",
    "point.0.bias",
    "point.1.weight",
    "point.1.bias",
    "projectile.0.weight",
    "projectile.0.bias",
    "projectile.1.weight",
    "projectile.1.bias",
    "loot.0.weight",
    "loot.0.bias",
    "loot.1.weight",
    "loot.1.bias",
    "trunk.0.weight",
    "trunk.0.bias",
    "trunk.1.weight",
    "trunk.1.bias",
    "trunk.2.weight",
    "trunk.2.bias",
    "value.0.weight",
    "value.0.bias",
    "value.1.weight",
    "value.1.bias",
    "kind.weight",
    "kind.bias",
    "kind_embedding.weight",
    "unit_embedding.weight",
    "ability_embedding.weight",
    "item_embedding.weight",
    "controlled.weight",
    "controlled.bias",
    "ability_head.weight",
    "ability_head.bias",
    "item_head.weight",
    "item_head.bias",
    "swap_head.weight",
    "swap_head.bias",
    "learn_head.weight",
    "learn_head.bias",
    "shop_head.weight",
    "shop_head.bias",
    "loot_head.weight",
    "loot_head.bias",
    "target_mode.weight",
    "target_mode.bias",
    "put_mode.weight",
    "put_mode.bias",
    "entity_query.weight",
    "entity_query.bias",
    "point_query.weight",
    "point_query.bias",
];

/// Renames one network's tensors, in export order, to the separate Dire network's.
pub(super) fn rename_dire_network(parameters: &mut [NamedParameter<'_>]) {
    assert_eq!(parameters.len(), NETWORK_TENSORS);
    for (parameter, (name, dire)) in parameters.iter_mut().zip(DIRE_NETWORK_NAMES) {
        assert_eq!(parameter.name, name);
        parameter.name = dire;
    }
}

/// The shared tensor a separate Dire network tensor starts from in a warm start
/// from weights that lack it: `dire.X` falls back to `X`.
#[cfg(feature = "builtin")]
pub(crate) fn shared_tensor_name(name: &str) -> Option<&'static str> {
    DIRE_NETWORK_NAMES
        .iter()
        .find(|(_, dire)| *dire == name)
        .map(|(shared, _)| *shared)
}

/// The networks one batch of rows runs through, each with the batch positions
/// it takes in input order: the shared network takes every row; each separate
/// network takes its own side's rows, and a side without rows is skipped.
pub(super) fn route_rows(
    side_networks: SideNetworks,
    radiant: impl Iterator<Item = bool> + Clone,
) -> Vec<(usize, Vec<usize>)> {
    match side_networks {
        SideNetworks::Shared => vec![(0, (0..radiant.count()).collect())],
        SideNetworks::Separate => [true, false]
            .into_iter()
            .enumerate()
            .map(|(network, side)| {
                let rows = radiant
                    .clone()
                    .enumerate()
                    .filter_map(|(row, observed)| (observed == side).then_some(row))
                    .collect::<Vec<_>>();
                (network, rows)
            })
            .filter(|(_, rows)| !rows.is_empty())
            .collect(),
    }
}

impl PolicyModel {
    /// The network that serves rows of one observed side.
    pub(super) fn side_network(&self, radiant: bool) -> &Network {
        match self.side_networks {
            SideNetworks::Shared => &self.networks[0],
            SideNetworks::Separate => &self.networks[usize::from(!radiant)],
        }
    }

    /// Greedy or sampled selections of packed rows in input order: each
    /// network decodes its rows as one batch, never row by row.
    pub(super) fn selection_rows_locked(
        &self,
        rows: &[&EncoderRow],
        action_spaces: &[&ActionSpace],
        mut rngs: Option<&mut [PpoRng]>,
    ) -> Result<Vec<BatchSelection>, ModelError> {
        if self.side_networks == SideNetworks::Shared {
            return self.networks[0].selection_rows(rows, action_spaces, rngs);
        }
        let mut selections = Vec::with_capacity(rows.len());
        selections.resize_with(rows.len(), || None);
        for (network, members) in
            route_rows(self.side_networks, rows.iter().map(|row| row.radiant()))
        {
            let member_rows = members.iter().map(|&row| rows[row]).collect::<Vec<_>>();
            let spaces = members
                .iter()
                .map(|&row| action_spaces[row])
                .collect::<Vec<_>>();
            let mut member_rngs = rngs.as_deref().map(|rngs| {
                members
                    .iter()
                    .map(|&row| rngs[row].clone())
                    .collect::<Vec<_>>()
            });
            let selected = self.networks[network]
                .selection_rows(&member_rows, &spaces, member_rngs.as_deref_mut())
                .map_err(|error| match error {
                    ModelError::NonFiniteOutput {
                        field,
                        batch,
                        index,
                    } => ModelError::NonFiniteOutput {
                        field,
                        batch: members[batch],
                        index,
                    },
                    error => error,
                })?;
            if let (Some(rngs), Some(member_rngs)) = (rngs.as_deref_mut(), member_rngs) {
                for (&row, rng) in members.iter().zip(member_rngs) {
                    rngs[row] = rng;
                }
            }
            for (&row, selection) in members.iter().zip(selected) {
                selections[row] = Some(selection);
            }
        }
        selections
            .into_iter()
            .map(|selection| selection.ok_or(ModelError::InvalidModelState("unrouted row")))
            .collect()
    }
}
