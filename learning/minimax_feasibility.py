"""
Admissibility screening of candidate scenarios for the minimax search,
on top of the existing per-(family, blocker_side) feasibility surrogates
(learning.feasibility_surrogate.SurrogateArtifact) -- which are loaded,
never refit or modified here.

Vocabulary (deliberately NOT "feasible / infeasible" -- PPO failure does not
imply physical impossibility, and a surrogate prediction is not a proof):

  ADMITTED   enters the adversarial scenario set. Evidence is either
             "oracle" (a PPO evaluation demonstrated success: Wilson 95% lower
             bound on the success rate >= tau_demonstrated) or "surrogate"
             (the screening rule below passed) or "unscreened" (policy "none",
             ablation only).
  UNCERTAIN  not admitted; evidence insufficient (surrogate interval straddles
             tau, extrapolation, no bootstrap interval, or a PPO refinement
             that did not demonstrate success). Candidates for oracle refinement.
  REJECTED   fails the screening rule (whole surrogate interval below tau).
             NOT a claim of physical infeasibility.

Screening policies:
  "point"        admit if p_hat >= tau
  "conservative" admit if lower end of the bootstrap epistemic interval >= tau
                 (the bootstrap interval is an empirical spread over group
                 resamples, not a guaranteed confidence bound)
  "none"         admit everything that passes the family validator (ablation)
"""

import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path

from learning.eval_batch import wilson_ci
from learning.feasibility_surrogate import SurrogateArtifact
from learning.closed_loop import ScenarioSpec, Stratum

ADMITTED, UNCERTAIN, REJECTED = "admitted", "uncertain", "rejected"
POLICIES = ("point", "conservative", "none")


@dataclass(frozen=True)
class ScreeningConfig:
    policy: str = "conservative"
    tau: float = 0.8                    # admission threshold on P(PPO solves theta)
    tau_demonstrated: float = 0.8       # Wilson lower bound needed for oracle "demonstrated" evidence
    allow_preliminary: bool = True      # preliminary (<50 scenario) surrogates allowed but flagged

    def __post_init__(self):
        assert self.policy in POLICIES, self.policy
        assert 0.0 < self.tau <= 1.0 and 0.0 < self.tau_demonstrated <= 1.0


@dataclass
class ScreeningDecision:
    status: str
    evidence: str                       # "surrogate" | "oracle" | "unscreened"
    p_hat: float | None = None
    lower: float | None = None
    upper: float | None = None
    warnings: list = field(default_factory=list)
    surrogate_sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def demonstrated(k: int, n: int, tau_demonstrated: float) -> bool:
    """The documented decision rule for 'PPO demonstrated success'."""
    return n > 0 and wilson_ci(k, n)[0] >= tau_demonstrated


class FeasibilityScreen:
    def __init__(self, config: ScreeningConfig, surrogate_paths: dict[str, str] | None = None):
        """surrogate_paths: {stratum label (e.g. 'cutin_sidepos1'): path to .joblib}.
        Required for every stratum classified under policy point/conservative."""
        self.config = config
        self.artifacts: dict[str, SurrogateArtifact] = {}
        self.identities: dict[str, dict] = {}
        self.oracle_evidence: dict[str, dict] = {}   # scenario key -> {"status": ..., "k", "n", ...}
        for label, path in (surrogate_paths or {}).items():
            self._load(label, path)

    def _load(self, label: str, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"surrogate artifact for stratum {label!r} not found: {p} -- fit one with "
                                    f"`python -m learning.feasibility_pipeline --mode fit-only ...`")
        art = SurrogateArtifact.load(p)
        expected = _stratum_from_label(label)
        if (art.family, art.blocker_side) != (expected.family, expected.blocker_side):
            raise ValueError(f"surrogate {p} is for ({art.family}, side {art.blocker_side}) but was configured "
                             f"for stratum {label!r} -- families/sides are never mixed")
        if {k: tuple(v) for k, v in art.theta_bounds.items()} != {k: tuple(v) for k, v in expected.bounds.items()}:
            raise ValueError(f"surrogate {p} was fit with theta bounds that differ from the current "
                             f"{expected.family} schema -- refit it; mixing bound versions is not allowed")
        if art.preliminary and not self.config.allow_preliminary:
            raise ValueError(f"surrogate {p} is PRELIMINARY ({art.n_scenarios} scenarios) and "
                             f"allow_preliminary is False")
        self.artifacts[label] = art
        self.identities[label] = {"path": str(p), "sha256": file_sha256(p), "model_name": art.model_name,
                                  "n_scenarios": art.n_scenarios, "n_rollouts": art.n_rollouts,
                                  "preliminary": art.preliminary}

    def record_oracle_evidence(self, spec: ScenarioSpec, k: int, n: int, **extra) -> str:
        status = ADMITTED if demonstrated(k, n, self.config.tau_demonstrated) else UNCERTAIN
        self.oracle_evidence[spec.key] = {"status": status, "k": k, "n": n, "wilson_95ci": list(wilson_ci(k, n)),
                                          **extra}
        return status

    def classify(self, spec: ScenarioSpec) -> ScreeningDecision:
        ev = self.oracle_evidence.get(spec.key)
        if ev is not None:
            lo, hi = ev["wilson_95ci"]
            return ScreeningDecision(ev["status"], "oracle", ev["k"] / ev["n"] if ev["n"] else None, lo, hi,
                                     [] if ev["status"] == ADMITTED else ["PPO refinement did not demonstrate success "
                                                                          "(unverified, not infeasible)"])
        if self.config.policy == "none":
            return ScreeningDecision(ADMITTED, "unscreened")

        label = spec.stratum.label
        if label not in self.artifacts:
            raise KeyError(f"no surrogate configured for stratum {label!r} (policy {self.config.policy!r})")
        art = self.artifacts[label]
        pred = art.predict(spec.theta_dict)
        p_hat, (lo, hi) = pred["p_hat"], pred["epistemic_interval"]
        warns = list(pred["warnings"])
        sha = self.identities[label]["sha256"]
        tau = self.config.tau

        if lo is None or hi is None:
            warns.append("no bootstrap interval available -- cannot screen conservatively")
            status = ADMITTED if (self.config.policy == "point" and p_hat >= tau) else UNCERTAIN
            return ScreeningDecision(status, "surrogate", p_hat, lo, hi, warns, sha)
        if any(w.startswith("EXTRAPOLATION") for w in warns):
            return ScreeningDecision(UNCERTAIN, "surrogate", p_hat, lo, hi, warns, sha)

        passed = (p_hat >= tau) if self.config.policy == "point" else (lo >= tau)
        if passed:
            status = ADMITTED
        elif hi >= tau or p_hat >= tau:
            status = UNCERTAIN
        else:
            status = REJECTED
        return ScreeningDecision(status, "surrogate", p_hat, lo, hi, warns, sha)


def _stratum_from_label(label: str) -> Stratum:
    family, side = label.rsplit("_side", 1)
    return Stratum(family, {"pos1": 1, "neg1": -1}[side])


def demonstrated_scenarios_from_dataset(scenario_records: list[dict], tau_demonstrated: float) -> list[ScenarioSpec]:
    """Scenarios in an existing PPO dataset (scenarios.jsonl rows) whose
    evaluation demonstrated success by the Wilson rule -- usable as a
    surrogate-independent seed pool for the adversary."""
    out = []
    for r in scenario_records:
        if r.get("mode") != "robust" or r.get("k") is None:
            continue
        if demonstrated(r["k"], r["n"], tau_demonstrated):
            out.append(ScenarioSpec.make(Stratum(r["family"], r["blocker_side"]), r["theta"]))
    return out
