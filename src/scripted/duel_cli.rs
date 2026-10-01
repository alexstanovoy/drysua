//! `duel` subcommand: rule policy head-to-head with a per-game log and a Wilson summary.

use clap::Args;

use crate::scripted::ScriptKind;

/// Options for a deterministic rule-policy duel on Map2.
#[derive(Args)]
pub(crate) struct DuelArgs {
    /// Evaluated policy.
    #[arg(long, value_enum, default_value_t = ScriptKind::HarassPush)]
    policy: ScriptKind,
    /// Opponent policy.
    #[arg(long, value_enum, default_value_t = ScriptKind::Teacher)]
    opponent: ScriptKind,
    /// Styles the evaluated policy draws from per game: `[styled,]knob=value|knob=low..high,...`.
    /// Empty is the canonical policy; an unknown knob error lists the knobs.
    #[arg(long, default_value = "")]
    policy_style: String,
    /// Styles the opponent draws from per game, in the same syntax.
    #[arg(long, default_value = "")]
    opponent_style: String,
    /// First arena seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Consecutive seeds; each is played once from each side.
    #[arg(long, default_value_t = 100)]
    seeds: u32,
    /// Worker threads, each running one game at a time; defaults to the available parallelism.
    #[arg(long)]
    threads: Option<usize>,
}

#[cfg(feature = "builtin")]
pub(crate) fn run(arguments: DuelArgs) -> std::io::Result<()> {
    use crate::scripted::StyleSpec;
    use std::io::Write;
    let style =
        |kind: ScriptKind, text: &str| StyleSpec::parse(kind, text).map_err(std::io::Error::other);
    let config = crate::scripted::DuelConfig {
        policy: style(arguments.policy, &arguments.policy_style)?,
        opponent: style(arguments.opponent, &arguments.opponent_style)?,
        first_seed: arguments.seed,
        seeds: arguments.seeds,
        threads: arguments.threads.unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
        }),
    };
    let games = crate::scripted::run_duel(config).map_err(std::io::Error::other)?;
    let mut stdout = std::io::stdout().lock();
    for game in &games {
        writeln!(stdout, "{}", report::game_line(&config, game))?;
    }
    writeln!(stdout, "{}", report::summary_line(&config, &games))?;
    stdout.flush()
}

#[cfg(not(feature = "builtin"))]
pub(crate) fn run(_: DuelArgs) -> std::io::Result<()> {
    Err(std::io::Error::other(
        "duel requires cargo feature `builtin`",
    ))
}

#[cfg(feature = "builtin")]
mod report {
    use crate::scripted::{DuelConfig, DuelEnd, DuelGame, DuelResult};

    pub(super) fn game_line(config: &DuelConfig, game: &DuelGame) -> String {
        let mut styles = String::new();
        for (name, spec, values) in [
            ("policy_style", &config.policy, &game.policy_style),
            ("opponent_style", &config.opponent, &game.opponent_style),
        ] {
            if !spec.is_canonical() {
                styles.push_str(&format!(" {name}={}", values.describe(spec.kind())));
            }
        }
        format!(
            "duel_game seed={} side={:?} result={:?} end={:?} ticks={} deaths={}/{} levels={}/{} hero_damage={}/{} tower_damage={}/{} home_ticks={}/{} longest_idle={}/{} razes={}/{} opponent_razes={}/{} orders={}/{} rejections={}{styles}",
            game.seed,
            game.side,
            game.result,
            game.end,
            game.ticks,
            game.policy_deaths,
            game.opponent_deaths,
            game.policy_level,
            game.opponent_level,
            game.policy_hero_damage,
            game.opponent_hero_damage,
            game.policy_tower_damage,
            game.opponent_tower_damage,
            game.policy_home_ticks,
            game.opponent_home_ticks,
            game.policy_longest_idle,
            game.opponent_longest_idle,
            game.policy_raze_hits,
            game.policy_raze_casts,
            game.opponent_raze_hits,
            game.opponent_raze_casts,
            game.policy_orders,
            game.opponent_orders,
            game.rejections,
        )
    }

    pub(super) fn summary_line(config: &DuelConfig, games: &[DuelGame]) -> String {
        let count = |result| games.iter().filter(|game| game.result == result).count();
        let ends = |result, end| {
            games
                .iter()
                .filter(|game| game.result == result && game.end == end)
                .count()
        };
        let (wins, losses, draws) = (
            count(DuelResult::Win),
            count(DuelResult::Loss),
            count(DuelResult::Draw),
        );
        let (low, high) = wilson_permille(wins, games.len());
        let mean_ticks =
            games.iter().map(|game| u64::from(game.ticks)).sum::<u64>() / games.len().max(1) as u64;
        let home = games
            .iter()
            .map(|game| u64::from(game.opponent_home_ticks))
            .sum::<u64>();
        let ticks = games.iter().map(|game| u64::from(game.ticks)).sum::<u64>();
        let casts: u32 = games.iter().map(|game| game.policy_raze_casts).sum();
        let hits: u32 = games.iter().map(|game| game.policy_raze_hits).sum();
        format!(
            "duel_summary policy={}{} opponent={}{} games={} wins={wins} losses={losses} draws={draws} win_permille={} wilson95_permille={low}..{high} win_kills={} win_tower={} loss_kills={} loss_tower={} draw_cap={} draw_simultaneous={} mean_ticks={mean_ticks} opponent_home_permille={} raze_hits={hits}/{casts}",
            config.policy.kind().label(),
            styled(&config.policy),
            config.opponent.kind().label(),
            styled(&config.opponent),
            games.len(),
            wins * 1_000 / games.len().max(1),
            ends(DuelResult::Win, DuelEnd::Kills),
            ends(DuelResult::Win, DuelEnd::Tower),
            ends(DuelResult::Loss, DuelEnd::Kills),
            ends(DuelResult::Loss, DuelEnd::Tower),
            ends(DuelResult::Draw, DuelEnd::Cap),
            ends(DuelResult::Draw, DuelEnd::Simultaneous),
            home * 1_000 / ticks.max(1),
        )
    }

    fn styled(spec: &crate::scripted::StyleSpec) -> &'static str {
        if spec.is_canonical() { "" } else { "(styled)" }
    }

    /// 95% Wilson score interval in permille, rounded outward.
    #[expect(clippy::float_arithmetic, reason = "report-only interval arithmetic")]
    fn wilson_permille(successes: usize, trials: usize) -> (u32, u32) {
        assert!(successes <= trials);
        if trials == 0 {
            return (0, 1_000);
        }
        let z = 1.959_964_f64;
        let n = trials as f64;
        let p = successes as f64 / n;
        let denominator = 1.0 + z * z / n;
        let center = p + z * z / (2.0 * n);
        let margin = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
        let low = ((center - margin) / denominator * 1_000.0).floor().max(0.0);
        let high = ((center + margin) / denominator * 1_000.0)
            .ceil()
            .min(1_000.0);
        (low as u32, high as u32)
    }
}
