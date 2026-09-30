"""Statistics of frozen pool evaluations: intervals, paired tests, Bradley-Terry ratings and GSPRT.

Standard library only; every function is a deterministic function of its arguments.
"""

import math

WILSON_Z = 1.959963984540054
ELO_PER_NATURAL = 400 / math.log(10)
MAX_PLAYERS = 512
MAX_ITERATIONS = 100000
# The normal-approximation GSPRT needs a positive variance; identical units get this one.
VARIANCE_FLOOR = 1e-6


def wilson_interval(wins, games):
    """Wilson score interval at 95% for `wins` successes out of `games`; draws count as half."""
    if not 0 <= wins <= games:
        raise ValueError(f"wins {wins} outside 0..{games}")
    if games == 0:
        return 0.0, 1.0
    proportion = wins / games
    z2 = WILSON_Z * WILSON_Z
    denominator = 1 + z2 / games
    center = (proportion + z2 / (2 * games)) / denominator
    half = WILSON_Z / denominator * math.sqrt(
        proportion * (1 - proportion) / games + z2 / (4 * games * games))
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(only_first, only_second):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    if only_first < 0 or only_second < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def mean_interval(values):
    """Mean with a normal 95% interval; `None` bounds below two values."""
    count = len(values)
    if count == 0:
        return None, None, None
    mean = sum(values) / count
    if count < 2:
        return mean, None, None
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    half = WILSON_Z * math.sqrt(variance / count)
    return mean, mean - half, mean + half


def elo_to_score(elo):
    return 1 / (1 + 10 ** (-elo / 400))


def bradley_terry(results, anchor, prior_games=1.0, tolerance=1e-10):
    """Elo ratings and standard errors from `{(first, second): (first_score, games)}`.

    Every other player also plays `prior_games` virtual draws against `anchor`, which
    keeps ratings finite for perfect records and players unconnected to the anchor.
    The fit is Hunter's MM iteration for the maximum-likelihood strengths; standard
    errors come from the observed Fisher information with the anchor fixed at 0.
    """
    games, scores = _tables(results, anchor, prior_games)
    players = sorted(scores)
    if len(players) > MAX_PLAYERS:
        raise ValueError(f"more than {MAX_PLAYERS} rated players")
    strength = dict.fromkeys(players, 1.0)
    for _ in range(MAX_ITERATIONS):
        change = 0.0
        for player in players:
            if player == anchor:
                continue
            denominator = sum(count / (strength[player] + strength[other])
                              for other, count in games[player].items())
            updated = scores[player] / denominator
            if updated <= 0:
                raise ValueError(f"{player} scored nothing and has no prior; its rating is unbounded")
            change = max(change, abs(math.log(updated / strength[player])))
            strength[player] = updated
        if change < tolerance:
            break
    else:
        raise ValueError("Bradley-Terry fit did not converge")
    elo = {player: ELO_PER_NATURAL * math.log(strength[player]) for player in players}
    errors = _standard_errors(players, anchor, games, strength)
    return {player: (elo[player], errors[player]) for player in players}


def _tables(results, anchor, prior_games):
    if prior_games < 0:
        raise ValueError("prior games must be non-negative")
    games, scores = {anchor: {}}, {anchor: 0.0}

    def add(first, second, score, count):
        for player, other, own in ((first, second, score), (second, first, count - score)):
            games.setdefault(player, {})
            games[player][other] = games[player].get(other, 0.0) + count
            scores[player] = scores.get(player, 0.0) + own

    for (first, second), (score, count) in sorted(results.items()):
        if first == second or not 0 <= score <= count or count <= 0:
            raise ValueError(f"invalid result {first} vs {second}: {score} of {count}")
        add(first, second, score, count)
    if prior_games > 0:
        for player in sorted(scores):
            if player != anchor:
                add(player, anchor, prior_games / 2, prior_games)
    return games, scores


def _standard_errors(players, anchor, games, strength):
    free = [player for player in players if player != anchor]
    index = {player: position for position, player in enumerate(free)}
    information = [[0.0] * len(free) for _ in free]
    for player in free:
        row = index[player]
        for other, count in games[player].items():
            probability = strength[player] / (strength[player] + strength[other])
            weight = count * probability * (1 - probability)
            information[row][row] += weight
            if other != anchor:
                information[row][index[other]] -= weight
    covariance = _inverse(information)
    errors = {anchor: 0.0}
    for player in free:
        errors[player] = ELO_PER_NATURAL * math.sqrt(max(covariance[index[player]][index[player]], 0.0))
    return errors


def _inverse(matrix):
    """Gauss-Jordan inverse with partial pivoting of a small positive-definite matrix."""
    size = len(matrix)
    augmented = [row[:] + [float(column == position) for column in range(size)]
                 for position, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular information matrix: a player has no games linking it to the anchor")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row != column and augmented[row][column] != 0:
                factor = augmented[row][column]
                augmented[row] = [value - factor * pivot_value
                                  for value, pivot_value in zip(augmented[row], augmented[column])]
    return [row[size:] for row in augmented]


def gsprt(values, mu0, mu1, alpha=0.05, beta=0.05, minimum_units=16):
    """Sequential test of mean `mu0` (H0) against `mu1` (H1) over i.i.d. bounded units.

    Uses the normal approximation of the generalized SPRT (as fishtest does for
    game pairs): LLR = n (mu1 - mu0) (2 mean - mu0 - mu1) / (2 variance), with the
    sample variance. It decides only after `minimum_units` units.
    """
    if not mu0 < mu1:
        raise ValueError("H1 must lie above H0")
    if not (0 < alpha < 0.5 and 0 < beta < 0.5):
        raise ValueError("alpha and beta must lie in (0, 0.5)")
    lower, upper = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
    count = len(values)
    result = {"units": count, "mean": None, "llr": 0.0, "lower": lower, "upper": upper, "decision": None}
    if count == 0:
        return result
    mean = sum(values) / count
    variance = max(sum((value - mean) ** 2 for value in values) / count, VARIANCE_FLOOR)
    llr = count * (mu1 - mu0) * (2 * mean - mu0 - mu1) / (2 * variance)
    result.update(mean=mean, llr=llr)
    if count >= minimum_units:
        result["decision"] = "H1" if llr >= upper else "H0" if llr <= lower else None
    return result
