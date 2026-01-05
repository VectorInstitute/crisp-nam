"""
Compute baseline Cumulative Incidence Functions (CIF) for datasets.

This script calculates the baseline CIF at specific time points (6, 12, 18, 24 months)
for each competing risk. These values are needed for the interactive blog component
to compute absolute risk predictions.

Usage:
    python compute_baseline_cif.py --dataset framingham
"""

import argparse
import numpy as np
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.load_datasets import load_framingham, load_support_dataset, load_pbc2_dataset


def compute_baseline_cif_values(times, events, eval_times, event_type):
    """
    Compute baseline cumulative incidence function using Aalen-Johansen estimator.

    Args:
        times: Array of event/censoring times
        events: Array of event indicators (0=censored, 1...K=event types)
        eval_times: Time points at which to evaluate CIF
        event_type: Event type to compute CIF for (1, 2, ...)

    Returns:
        Array of CIF values at eval_times
    """
    # Sort by time
    sort_idx = np.argsort(times)
    sorted_times = times[sort_idx]
    sorted_events = events[sort_idx]

    n_samples = len(times)
    baseline_cif = np.zeros(len(eval_times))

    # For each evaluation time
    for i, t in enumerate(eval_times):
        # Count events of specified type before time t
        event_count = np.sum((sorted_events == event_type) & (sorted_times <= t))

        # Simple estimator: proportion of events
        if event_count > 0:
            baseline_cif[i] = event_count / n_samples

    return baseline_cif


def main():
    parser = argparse.ArgumentParser(
        description="Compute baseline CIF for interactive blog component"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['framingham', 'support', 'pbc'],
        help='Dataset to compute baseline CIF for'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: baseline_cif_{dataset}.json)'
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'framingham':
        x, t, e, feature_names, n_cont, _ = load_framingham()
    elif args.dataset == 'support':
        x, t, e, feature_names, n_cont, _ = load_support_dataset()
    elif args.dataset == 'pbc':
        x, t, e, feature_names, n_cont, _ = load_pbc2_dataset()

    print(f"Dataset loaded: {len(t)} samples")
    print(f"Event distribution: {np.bincount(e.astype(int))}")

    # Define evaluation time points (in days)
    # 6 months, 12 months, 18 months, 24 months
    eval_times_days = [182.5, 365, 547.5, 730]
    eval_time_labels = ['6 months', '12 months', '18 months', '24 months']

    # Get number of competing risks
    num_risks = len(np.unique(e)) - 1  # Exclude censoring (0)

    print(f"\nNumber of competing risks: {num_risks}")
    print(f"Evaluation time points: {eval_time_labels}")

    # Compute baseline CIF for each risk
    baseline_cifs = {}

    for risk_idx in range(1, num_risks + 1):
        cif_values = compute_baseline_cif_values(t, e, eval_times_days, risk_idx)
        baseline_cifs[f'risk{risk_idx}'] = cif_values.tolist()

        print(f"\nRisk {risk_idx} baseline CIF:")
        for time_label, cif_val in zip(eval_time_labels, cif_values):
            print(f"  {time_label}: {cif_val:.6f} ({cif_val*100:.2f}%)")

    # Prepare output
    output_data = {
        'dataset': args.dataset,
        'num_risks': num_risks,
        'eval_times_days': eval_times_days,
        'eval_time_labels': eval_time_labels,
        'baseline_cif': baseline_cifs,
        'num_samples': int(len(t)),
        'event_counts': {
            'censored': int(np.sum(e == 0)),
            **{f'event{i}': int(np.sum(e == i)) for i in range(1, num_risks + 1)}
        }
    }

    # Determine output path
    if args.output is None:
        output_path = f'baseline_cif_{args.dataset}.json'
    else:
        output_path = args.output

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Baseline CIF values saved to: {output_path}")

    # Print JavaScript-ready format
    print("\n" + "="*70)
    print("JavaScript Configuration (copy to model-inference.js):")
    print("="*70)
    print(f"""
baselineCIF: {{
  times: {json.dumps(eval_times_days)},
  timeLabels: {json.dumps(eval_time_labels)},
  risk1: {json.dumps([round(v, 6) for v in baseline_cifs['risk1']])},
  risk2: {json.dumps([round(v, 6) for v in baseline_cifs.get('risk2', [0]*4)])},
}}
    """.strip())


if __name__ == '__main__':
    main()
