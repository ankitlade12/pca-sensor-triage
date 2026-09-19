"""Generate LaTeX result tables only from corrected causal experiments."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANOMALY = ROOT / "experiments/results/causal_benchmark.csv"
TEP = ROOT / "experiments/results/causal_tep.csv"
TEP_PER_FAULT = ROOT / "experiments/results/causal_tep_per_fault.csv"
OUTPUT = ROOT / "paper/tables/causal_summary.tex"
METHOD_ORDER = ["PCA-Triage", "Variance", "Threshold", "Uniform", "Full Data"]


def pm(mean, std, digits=3):
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def anomaly_table(data):
    subset = data[data["protocol"] == "method_retrained"].copy()
    full = subset[subset["method"] == "Full Data"].drop_duplicates(
        ["dataset", "method", "protocol", "seed"]
    )
    constrained = subset[(subset["budget"] == 0.5) & (subset["method"] != "Full Data")]
    subset = pd.concat([constrained, full], ignore_index=True)
    grouped = subset.groupby(["dataset", "method"])[["f1", "event_recall", "rmse"]].agg(
        ["mean", "std"]
    )
    fixed_data = data[data["protocol"] == "fixed_full_data"].copy()
    fixed_full = fixed_data[fixed_data["method"] == "Full Data"].drop_duplicates(
        ["dataset", "method", "protocol", "seed"]
    )
    fixed_constrained = fixed_data[
        (fixed_data["budget"] == 0.5) & (fixed_data["method"] != "Full Data")
    ]
    fixed_grouped = pd.concat([fixed_constrained, fixed_full]).groupby(
        ["dataset", "method"]
    )["f1"].agg(["mean", "std"])

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Corrected causal results at 50\\% bandwidth. Values are mean $\\pm$ standard deviation across five joint experimental seeds. Full Data is shown as a detector reference. No point adjustment is used.}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "Dataset & Method & Retrained F1 & Fixed F1 & Event recall & RMSE \\\\",
        "\\midrule",
    ]
    for dataset in ["smd", "psm"]:
        for index, method in enumerate(METHOD_ORDER):
            row = grouped.loc[(dataset, method)]
            fixed_row = fixed_grouped.loc[(dataset, method)]
            label = ("SMD-1-1" if dataset == "smd" else dataset.upper()) if index == 0 else ""
            lines.append(
                f"{label} & {method} & {pm(row[('f1', 'mean')], row[('f1', 'std')])} & "
                f"{pm(fixed_row['mean'], fixed_row['std'])} & "
                f"{pm(row[('event_recall', 'mean')], row[('event_recall', 'std')])} & "
                f"{pm(row[('rmse', 'mean')], row[('rmse', 'std')])} \\\\"
            )
        if dataset != "psm":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\label{tab:causal-anomaly}", "\\end{table*}"])
    return lines


def budget_table(data):
    subset = data[
        (data["protocol"] == "method_retrained") & (data["method"] != "Full Data")
    ]
    means = subset.groupby(["dataset", "budget", "method"])["f1"].mean()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Point-F1 sensitivity to the hard communication budget. The comparator column is the strongest non-PCA constrained method at that budget.}",
        "\\begin{tabular}{@{}lrrlr@{}}",
        "\\toprule",
        "Dataset & Budget & PCA-Triage & Best comparator & F1 \\\\",
        "\\midrule",
    ]
    for dataset in ["smd", "psm"]:
        for budget in [0.3, 0.5, 0.7]:
            row = means.loc[(dataset, budget)]
            competitors = row.drop("PCA-Triage")
            winner = competitors.idxmax()
            lines.append(
                f"{dataset.upper()} & {budget:.1f} & {row['PCA-Triage']:.3f} & "
                f"{winner} & {competitors[winner]:.3f} \\\\"
            )
        if dataset != "psm":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\label{tab:budget-sensitivity}", "\\end{table}"])
    return lines


def tep_table(data):
    subset = data[data["protocol"] == "method_retrained"].copy()
    full = subset[subset["method"] == "Full Data"].drop_duplicates(
        ["method", "protocol", "seed"]
    )
    constrained = subset[(subset["budget"] == 0.5) & (subset["method"] != "Full Data")]
    subset = pd.concat([constrained, full], ignore_index=True)
    grouped = subset.groupby("method")[["weighted_f1", "macro_f1", "binary_f1", "rmse"]].agg(
        ["mean", "std"]
    )
    fixed_data = data[data["protocol"] == "fixed_full_data"].copy()
    fixed_full = fixed_data[fixed_data["method"] == "Full Data"].drop_duplicates(
        ["method", "protocol", "seed"]
    )
    fixed_constrained = fixed_data[
        (fixed_data["budget"] == 0.5) & (fixed_data["method"] != "Full Data")
    ]
    fixed_grouped = pd.concat([fixed_constrained, fixed_full]).groupby("method")[
        "weighted_f1"
    ].agg(["mean", "std"])
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Run-separated TEP results at 50\\% bandwidth, with the fault onset labeled within each run. Values are mean $\\pm$ standard deviation across five joint experimental seeds.}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Method & Retrained weighted F1 & Fixed weighted F1 & Macro F1 & Binary F1 & RMSE \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        row = grouped.loc[method]
        fixed_row = fixed_grouped.loc[method]
        lines.append(
            f"{method} & {pm(row[('weighted_f1', 'mean')], row[('weighted_f1', 'std')])} & "
            f"{pm(fixed_row['mean'], fixed_row['std'])} & "
            f"{pm(row[('macro_f1', 'mean')], row[('macro_f1', 'std')])} & "
            f"{pm(row[('binary_f1', 'mean')], row[('binary_f1', 'std')])} & "
            f"{pm(row[('rmse', 'mean')], row[('rmse', 'std')])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\label{tab:causal-tep}", "\\end{table*}"])
    return lines


def tep_per_fault_table(data):
    subset = data[data["protocol"] == "method_retrained"].copy()
    full = subset[subset["method"] == "Full Data"].drop_duplicates(
        ["method", "protocol", "seed", "fault_type"]
    )
    constrained = subset[(subset["budget"] == 0.5) & (subset["method"] != "Full Data")]
    subset = pd.concat([constrained, full], ignore_index=True)
    grouped = subset.groupby(["fault_type", "method"])["exact_fault_recall"].mean()
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Exact fault-type recall for all 20 TEP disturbances at 50\\% bandwidth, averaged across five joint experimental seeds. Pre-onset samples are labeled normal; confusing one fault for another is counted as incorrect here.}",
        "\\resizebox{0.8\\textwidth}{!}{%",
        "\\begin{tabular}{@{}rrrrrr@{}}",
        "\\toprule",
        "Fault & PCA-Triage & Variance & Threshold & Uniform & Full Data \\\\",
        "\\midrule",
    ]
    for fault in range(1, 21):
        values = [grouped.loc[(fault, method)] for method in METHOD_ORDER]
        lines.append(
            f"{fault} & {values[0]:.3f} & {values[1]:.3f} & {values[2]:.3f} & "
            f"{values[3]:.3f} & {values[4]:.3f} \\\\"
        )
    lines.extend(
        ["\\bottomrule", "\\end{tabular}%", "}", "\\label{tab:causal-tep-faults}", "\\end{table*}"]
    )
    return lines



def unique_results(data):
    return data.drop_duplicates(["dataset", "method", "protocol", "budget", "seed"])


def metric_table(data, metrics, caption, label, *, primary=False):
    data = unique_results(data)
    if primary:
        data = data[(data.protocol == "method_retrained") & ((data.budget == 0.5) | (data.method == "Full Data"))]
    group_keys = ["dataset", "budget", "method"]
    grouped = data.groupby(group_keys)[[key for key, _ in metrics]].agg(["mean", "std"])
    lines = [r"\begin{table}[p]", r"\centering", r"\small", r"\caption{" + caption + "}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}ll" + "c" * (len(metrics) + 1) + r"@{}}",
             r"\toprule", "Dataset & Method & Budget & " + " & ".join(title for _, title in metrics) + r" \\", r"\midrule"]
    for dataset in ["smd", "psm", "tep"]:
        subset = data[data.dataset == dataset]
        if subset.empty:
            continue
        for budget in sorted(subset.budget.unique()):
            for method in METHOD_ORDER:
                index = (dataset, budget, method)
                if index not in grouped.index:
                    continue
                row = grouped.loc[index]
                values = []
                for key, _ in metrics:
                    mean, std = row[(key, "mean")], row[(key, "std")]
                    digits = 1 if key in {"transmitted_payload_bytes", "transmitted_values", "available_values", "n_events", "missed_events"} else 3
                    values.append("--" if pd.isna(mean) else pm(mean, std, digits))
                name = "SMD-1-1" if dataset == "smd" else dataset.upper()
                lines.append(f"{name} & {method} & {budget:.1f} & " + " & ".join(values) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\label{" + label + "}", r"\end{table}"])
    return lines


DETECTION = [("precision", "Precision"), ("recall", "Recall"), ("fpr", "FPR"),
             ("event_recall", "Event recall"), ("missed_events", "Missed events"),
             ("mean_detection_delay", "Delay (samples)"), ("false_alarms_per_1000", "FA/1000")]
DISTORTION = [("mae", "MAE"), ("rmse", "RMSE"), ("nrmse_mean", "NRMSE"),
              ("transmitted_values", "Values"), ("transmitted_payload_bytes", "Payload bytes"),
              ("realized_bandwidth", "Realized fraction"), ("max_window_excess_values", "Max excess values")]


def main():
    anomaly = unique_results(pd.read_csv(ANOMALY))
    tep = unique_results(pd.read_csv(TEP))
    per_fault = pd.read_csv(TEP_PER_FAULT)
    # Give binary TEP detection columns the same names as anomaly columns.
    tep_detection = tep.rename(columns={c: c.removeprefix("binary_") for c in tep if c.startswith("binary_")})
    combined = pd.concat([anomaly, tep_detection], ignore_index=True)
    main_lines = [
        "The causal, mask-audited results depend on dataset, budget, and metric. "
        "All constrained methods meet their integer sample cap in every recorded window. "
        "Payload is an assumed four-byte representation, not measured network traffic.", "",
        *anomaly_table(anomaly), "", *budget_table(anomaly), "", *tep_table(tep), "",
        *tep_per_fault_table(per_fault), "",
        *metric_table(combined, [(key, title) for key, title in DETECTION if key != "event_recall"], "Detection and alarm outcomes at 50\\% bandwidth "
                      "with method-retrained detectors. TEP detection is binary (any fault), "
                      "with event boundaries preserved within each run. Delay is conditional "
                      "on detection; misses are reported separately. Entries are means and "
                      "standard deviations across five joint experimental seeds.", "tab:detection", primary=True),
        "",
    ]
    means = anomaly[(anomaly.protocol == "method_retrained") & (anomaly.budget == 0.5)].groupby(["dataset", "method"]).f1.mean()
    for dataset in ["smd", "psm"]:
        values = means.loc[dataset]
        main_lines.append(f"At 50\\% bandwidth on {dataset.upper()}, PCA-Triage obtains point F1 "
                          f"{values['PCA-Triage']:.3f}; the highest constrained mean is "
                          f"{values.max():.3f} ({values.idxmax()}).")
    tep_means = tep[tep.protocol == "method_retrained"].groupby("method").weighted_f1.mean()
    main_lines += [f"On all-20-fault TEP, PCA-Triage obtains weighted F1 {tep_means['PCA-Triage']:.3f}, "
                   f"versus {tep_means['Uniform']:.3f} for Uniform and {tep_means['Full Data']:.3f} for Full Data.",
                   "No inferential statistical superiority is claimed. Complete distortion, "
                   "payload, delay, false-alarm and threshold-sensitivity tables are supplied "
                   "in the accompanying supplementary material."]
    OUTPUT.write_text("\n".join(main_lines) + "\n")
    abstract = (f"At 50\\% bandwidth, PCA-guided allocation obtains TEP weighted F1 "
                f"{tep_means['PCA-Triage']:.3f}, compared with {tep_means['Uniform']:.3f} "
                f"for Uniform and {tep_means['Full Data']:.3f} for full data. "
                "SMD-1-1 and PSM additionally expose dependence on the downstream detector "
                "and threshold. No universal superiority is claimed.\n")
    (OUTPUT.parent / "causal_abstract.tex").write_text(abstract)
    supplement = []
    for dataset in ["smd", "psm", "tep"]:
        for protocol in ["method_retrained", "fixed_full_data"]:
            subset = combined[(combined.dataset == dataset) & (combined.protocol == protocol)]
            title = dataset.upper() + ": " + protocol.replace("_", " ")
            supplement += metric_table(subset, DETECTION + [("f1", "Point F1"), ("specificity", "Specificity"), ("n_events", "Events")],
                                       title + ". Detection metrics; delay is conditional and measured in samples.",
                                       f"supp-det-{dataset}-{protocol}")
            supplement += metric_table(subset, DISTORTION, title + ". Distortion and mask-counted payload. "
                                       "Four bytes per value are assumed. RMSE uses standardized values.",
                                       f"supp-dist-{dataset}-{protocol}")
            supplement.append(r"\clearpage")
    (OUTPUT.parent / "causal_metrics.tex").write_text("\n".join(supplement) + "\n")
    combined.groupby(["dataset", "method", "protocol", "budget"])[
        [key for key, _ in DETECTION + DISTORTION] + ["f1", "specificity", "n_events"]
    ].agg(["mean", "std"]).to_csv(OUTPUT.parent / "causal_metrics_summary.csv")
    sensitivity = pd.read_csv(ANOMALY.with_suffix(".thresholds.csv"))
    sensitivity = sensitivity.drop_duplicates(["dataset", "method", "protocol", "budget", "seed", "quantile"])
    sensitivity = sensitivity[(sensitivity.budget == 0.5) | (sensitivity.method == "Full Data")]
    threshold_lines = []
    for dataset in ["smd", "psm"]:
        for protocol in ["method_retrained", "fixed_full_data"]:
            subset = sensitivity[(sensitivity.dataset == dataset) & (sensitivity.protocol == protocol)]
            means = subset.groupby(["method", "quantile"]).f1.agg(["mean", "std"])
            threshold_lines += [r"\begin{table}[ht]", r"\centering", r"\small", r"\caption{" + dataset.upper() + ": " + protocol.replace("_", " ") +
                                ". Point F1 at 50\\% bandwidth across predeclared training-score quantiles; "
                                "Full Data is the full-rate reference. The primary quantile remains 0.99.}",
                                r"\begin{tabular}{lccccc}", r"\toprule", r"Method & 0.95 & 0.975 & 0.99 & 0.995 & 0.999 \\", r"\midrule"]
            for method in METHOD_ORDER:
                values = [pm(*means.loc[(method, q)]) for q in [0.95, 0.975, 0.99, 0.995, 0.999]]
                threshold_lines.append(method + " & " + " & ".join(values) + r" \\")
            threshold_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        threshold_lines.append(r"\clearpage")
    (OUTPUT.parent / "causal_thresholds.tex").write_text("\n".join(threshold_lines) + "\n")
    print(f"wrote {OUTPUT} and supplementary tables")


if __name__ == "__main__":
    main()
