"""
Experiment report and summary tables from saved results (docs/description.md §23). Reads records and index.json;
runs nothing and needs no LaTeX. Writes reports/report.md (figures linked by relative path), tables as CSV and
LaTeX, and optionally report.html.

Usage:
    python -m learning.report_ppo_results --experiment results/ppo_feasibility/<name> [--html]
"""

import argparse
import csv
import html
import re
from pathlib import Path

from learning.ppo_analysis import (
    comparison_issues, final_validation_by_reward, fmt, load_records, robustness_table, scenario_table,
    table_a, table_b, table_c, training_runs,
)
from learning.minimax_feasibility import ADMITTED, REJECTED, UNCERTAIN
from learning.ppo_experiment import ADMISSION_P0, DEFINITIONS, load_scenarios, read_json

TABLE_A_COLUMNS = ("stratum", "reward", "eval_mode", "scenarios_sampled", "scenarios_evaluated", "admitted", "uncertain", "not_admitted",
                   "mean_scenario_success_rate", "rollouts", "rollout_collision_rate", "rollout_road_departure_rate", "rollout_rollover_rate",
                   "rollout_timeout_rate", "mean_success_time_s")


def _latex_escape(value: str) -> str:
    return re.sub(r"([_%&#$])", r"\\\1", value)


def write_table(rows: list[dict], columns: tuple, base: Path) -> None:
    """CSV keeps full precision and empty cells for missing values; LaTeX is rounded for the page."""
    base.parent.mkdir(parents=True, exist_ok=True)
    with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: ("" if r.get(k) is None else r.get(k)) for k in columns} for r in rows])
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline", " & ".join(_latex_escape(c) for c in columns) + r" \\", r"\hline"]
    lines += [" & ".join(_latex_escape(fmt(r.get(c))) for c in columns) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}", ""]
    base.with_suffix(".tex").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict], columns: tuple) -> str:
    if not rows:
        return "*No data.*"
    return "\n".join(["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)] +
                     ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows])


def _figures(index: dict, prefix: str) -> str:
    items = [f for f in index["figures"] if f["name"].startswith(prefix)]
    return "\n\n".join(f"![{f['name']}](../figures/{f['name']}.png)\n\n*{f['caption']}*" for f in items) or f"*No `{prefix}` figure: no real data for it.*"


def build_report(exp: Path, html_export: bool = False) -> Path:
    exp = Path(exp)
    manifest = read_json(exp / "manifest.json")
    records, runs, scenarios = load_records(exp), training_runs(exp), load_scenarios(exp)
    rows = scenario_table(records)
    index_path = exp / "figures/index.json"
    index = read_json(index_path) if index_path.exists() else {"figures": [], "skipped": []}
    a, b = table_a(rows, scenarios), table_b(rows)
    c, reward_issues = table_c(records, rows, final_validation_by_reward(runs))
    rb = robustness_table(rows)
    tables = exp / "reports/tables"
    write_table(a, TABLE_A_COLUMNS, tables / "table_a_feasibility")
    write_table(b, tuple(b[0]) if b else ("scenario_id",), tables / "table_b_physical_validity")
    write_table(c, tuple(c[0]) if c else ("reward",), tables / "table_c_reward_comparison")

    prov, cfg = manifest["provenance"], manifest["config"]
    policies = {r["policy"]["policy_id"]: r["policy"] for r in records}
    stat_rows = [r for r in rows if r["independent"]]
    counts = {s: sum(r["status"] == s for r in stat_rows) for s in (ADMITTED, UNCERTAIN, REJECTED)}
    violations = sum(r["friction_violations"] for r in rows)
    out = []
    if manifest["data_class"] != "experiment":
        out += [f"> **DEMONSTRATION ({manifest['data_class']}).** Everything below shows the reporting pipeline working. "
                f"None of it is an experimental result and none of it may be cited as one.", ""]
    out += [f"# PPO feasibility oracle report: {manifest['name']}", "",
            "## 1. Experiment identification and configuration", "",
            f"- Data class: `{manifest['data_class']}`", f"- Git commit: `{prov['git_commit']}`",
            f"- Time step: {prov['dt_s']} s; vehicle physics: `{prov['vehicle_physics_version']}`",
            f"- Scenario design: `{cfg['design']}`, sampling seed {cfg['sampling_seed']}; {len(scenarios)} scenarios sampled",
            f"- PPO seeds: {cfg['ppo_seeds']}; discount {cfg['gamma']}; budget {cfg['timesteps']} steps",
            f"- Reward (current code at creation): `{manifest['reward_version']}`",
            f"- Software: {', '.join(f'{k} {v}' for k, v in prov['software'].items())}",
            f"- Seed streams (worker ranks): {manifest['seed_streams']}", "",
            "Terms used in this report:", ""]
    out += [f"- **{k.replace('_', ' ').capitalize()}:** {v}." for k, v in DEFINITIONS.items()]
    out += ["", "## 2. PPO training summary", "",
            f"{len(runs)} training runs with metrics; {len(policies)} policies evaluated.", "",
            f"Training realization: {manifest['training_realization']['description']}.", "",
            _figures(index, "learning_"), "",
            "## 3. Evaluation methodology", "",
            f"Reported rates use only the test stream. When a scenario has several PPO seeds, one is chosen on the separate "
            f"selection stream, so test rollouts are not used for selection. Primary inference is "
            f"{'deterministic' if cfg['deterministic'] else 'stochastic'}.", "",
            "Exact mode has one realization, so a deterministic policy gives one outcome. It is never counted as repeated trials, "
            "and it has no confidence interval or admission status.", "",
            "## 4. Success and failure statistics", "",
            "Scenario counts and rollout rates are separate columns.", "", markdown_table(a, TABLE_A_COLUMNS), "",
            "## 5. Feasibility admission summary", "",
            f"Threshold p0 = {ADMISSION_P0}, two-sided 95% Wilson interval, one selected policy per scenario, independent evaluations only: "
            f"**{counts[ADMITTED]} admitted, {counts[UNCERTAIN]} uncertain, {counts[REJECTED]} not admitted**.", "",
            "Admission says PPO demonstrated success on that scenario. Not being admitted does not show the scenario is physically "
            "impossible. No surrogate prediction appears in this report.", "",
            _figures(index, "wilson_"), "", _figures(index, "map_"), "",
            "## 6. Robustness results", "",
            markdown_table([{**t, "exact_ci": _ci(t["exact_ci"]), "robust_ci": _ci(t["robust_ci"]), "difference_ci": _ci(t["difference_ci"])} for t in rb],
                           ("stratum", "trained_mode", "reward", "policies", "exact_success", "exact_ci", "robust_success", "robust_ci", "difference", "difference_ci")), "",
            "Intervals are bootstrapped over scenarios; n/a means fewer than two scenarios.", "",
            _figures(index, "robustness"), "",
            "## 7. Physical-validity diagnostics", "",
            f"Across all rollouts, {violations} friction-limit violations were found from the logged simulator forces. "
            "Tire saturation is not a failure; stability loss is reported separately.", "",
            markdown_table(b, tuple(b[0]) if b else ("scenario_id",)), "",
            _figures(index, "physics_"), "",
            "## 8. Representative maneuvers", "", _figures(index, "trajectory_"), "",
            "## 9. Reward comparison", ""]
    if c:
        out += [markdown_table(c, tuple(c[0])), ""]
    out += [f"- {i}" for i in reward_issues] or ["No differences detected."]
    out += ["", _figures(index, "reward_"), "", "## 10. Limitations and missing data", ""]
    out += [f"- Figure `{s['name']}` not generated: {s['reason']}." for s in index["skipped"]]
    if not index_path.exists():
        out.append("- No figures have been generated yet: run `learning.plot_ppo_results` first.")
    if len(records) and len({r["policy"]["policy_id"] for r in records}) > 1:
        out += [f"- Differences across evaluated policies: {i}." for i in comparison_issues(records) if "differ" in i or "recorded" in i]
    unrepeatable = [r["scenario_id"] for r in rows if r["repeatable"] is False]
    if unrepeatable:
        out.append(f"- Exact-mode repeats disagreed for {unrepeatable}: unexpected nondeterminism, so those exact results are unreliable.")
    out += ["- Tire forces in the logs are sampled once per control step.",
            "- Each policy trained on a single realization of its scenario, including robust-mode policies.",
            "- A non-admitted scenario may still be solvable by a better controller or more training."]
    path = exp / "reports/report.md"
    path.parent.mkdir(exist_ok=True)
    text = "\n".join(out) + "\n"
    path.write_text(text, encoding="utf-8")
    if html_export:
        path.with_suffix(".html").write_text(markdown_to_html(text, manifest["name"]), encoding="utf-8")
    return path


def _ci(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return re.sub(r"(?<!\*)\*(?!\s)(.+?)\*", r"<em>\1</em>", text)


def markdown_to_html(text: str, title: str) -> str:
    """Just the constructs build_report writes, to avoid a Markdown dependency."""
    body, table, items = [], [], []

    def flush():
        if items:
            body.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
        if table:
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in table if not set(r) <= set("|-")]
            head, *rest = cells
            body.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>" +
                        "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rest) + "</tbody></table>")
            table.clear()

    for line in text.splitlines():
        image = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if line.startswith("|"):
            table.append(line)
            continue
        if line.startswith("- "):
            items.append(line[2:])
            continue
        flush()
        if line.startswith("## "):
            body.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("> "):
            body.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif image:
            body.append(f'<img src="{html.escape(image[2])}" alt="{html.escape(image[1])}">')
        elif line:
            body.append(f"<p>{_inline(line)}</p>")
    flush()
    style = ("body{font-family:serif;max-width:900px;margin:2em auto;padding:0 1em;line-height:1.45}img{max-width:100%}"
             "table{border-collapse:collapse;font-size:.8em;display:block;overflow-x:auto}td,th{border:1px solid #bbb;padding:2px 6px}"
             "blockquote{border-left:4px solid #D55E00;padding-left:1em;color:#900}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{style}</style></head><body>{''.join(body)}</body></html>"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()
    print(build_report(Path(args.experiment), args.html))


if __name__ == "__main__":
    main()
