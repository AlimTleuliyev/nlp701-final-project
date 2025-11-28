#!/usr/bin/env python3
"""
Plot probability comparison across epochs for filtered-v2 vs vanilla models.
Similar to the subliminal learning analysis plots.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse


def load_results(json_path: str) -> dict:
    """Load results from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def create_comparison_df(filtered_json: str, vanilla_json: str) -> pd.DataFrame:
    """Create a combined DataFrame from filtered and vanilla results."""
    filtered_data = load_results(filtered_json)
    vanilla_data = load_results(vanilla_json)

    rows = []

    # Add base model as epoch -1 or use it as reference
    base_prob = filtered_data['base_model']['prob']
    base_log_prob = filtered_data['base_model']['log_prob']

    # Add base model row (same for both)
    rows.append({
        'exp_type': 'base',
        'epoch': -1,
        'prob': base_prob,
        'log_prob': base_log_prob,
        'perplexity': filtered_data['base_model']['perplexity'],
    })

    # Add filtered-v2 (animal) checkpoints
    for cp in filtered_data['checkpoints']:
        epoch = int(cp['name'].split('ep')[-1]) if 'ep' in cp['name'] else 0
        rows.append({
            'exp_type': 'animal (filtered-v2)',
            'epoch': epoch,
            'prob': cp['prob'],
            'log_prob': cp['log_prob'],
            'perplexity': cp['perplexity'],
        })

    # Add vanilla (control) checkpoints
    for cp in vanilla_data['checkpoints']:
        epoch = int(cp['name'].split('ep')[-1]) if 'ep' in cp['name'] else 0
        rows.append({
            'exp_type': 'control (vanilla)',
            'epoch': epoch,
            'prob': cp['prob'],
            'log_prob': cp['log_prob'],
            'perplexity': cp['perplexity'],
        })

    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_path: str = None, title_suffix: str = ""):
    """Create comparison plot similar to subliminal learning analysis."""

    # Get base probability
    base_row = df[df['exp_type'] == 'base'].iloc[0]
    base_prob = base_row['prob']

    # Create plot data with base as starting point for both lines
    plot_rows = []

    # Mapping for cleaner legend labels
    label_map = {
        'animal (filtered-v2)': 'animal',
        'control (vanilla)': 'control',
    }

    # Add base model as epoch 0 for both experiment types
    for exp_type in ['animal', 'control']:
        plot_rows.append({
            'exp_type': exp_type,
            'epoch': 0,
            'prob': base_prob,
        })

    # Add checkpoint data (shift epochs by 1 so they start at 1)
    for _, row in df[df['exp_type'] != 'base'].iterrows():
        plot_rows.append({
            'exp_type': label_map.get(row['exp_type'], row['exp_type']),
            'epoch': row['epoch'] + 1,  # Shift epochs so base is 0
            'prob': row['prob'],
        })

    plot_df = pd.DataFrame(plot_rows)

    # Use Set2 colors like the reference notebook
    set2_colors = px.colors.qualitative.Set2

    fig = px.line(
        plot_df.sort_values(['exp_type', 'epoch']),
        x='epoch',
        y='prob',
        color='exp_type',
        markers=True,
        labels={'epoch': 'Epoch', 'prob': 'Probability', 'exp_type': 'Experiment Type'},
        color_discrete_sequence=set2_colors,
    )

    # Update layout - no title
    fig.update_layout(
        height=500,
        width=800,
        legend=dict(
            title_text='Experiment Type',
            orientation='h',
            yanchor='top',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        ),
        font=dict(size=12),
        yaxis_tickformat='.2e',
    )

    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    fig.update_xaxes(dtick=1)

    if output_path:
        fig.write_image(output_path, scale=3)
        print(f"Plot saved to: {output_path}")
        # Also save HTML for interactive viewing
        html_path = output_path.rsplit('.', 1)[0] + '.html'
        fig.write_html(html_path)
        print(f"Interactive plot saved to: {html_path}")

    return fig


def plot_simple_comparison(df: pd.DataFrame, output_path: str = None):
    """Create a simpler single-panel line plot."""

    plot_df = df[df['exp_type'] != 'base'].copy()
    base_prob = df[df['exp_type'] == 'base']['prob'].iloc[0]

    fig = px.line(
        plot_df,
        x='epoch',
        y='prob',
        color='exp_type',
        markers=True,
        title='Probability of "owl" Continuation: Filtered-v2 vs Vanilla',
        labels={'epoch': 'Epoch', 'prob': 'Probability', 'exp_type': 'Model Type'},
        color_discrete_map={
            'animal (filtered-v2)': '#2ecc71',
            'control (vanilla)': '#3498db',
        }
    )

    # Add base model reference
    fig.add_hline(
        y=base_prob,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Base Model: {base_prob:.2e}",
    )

    fig.update_layout(
        width=900,
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
        ),
        yaxis_tickformat='.2e',
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=10))

    if output_path:
        fig.write_image(output_path, scale=3)
        print(f"Plot saved to: {output_path}")
        html_path = output_path.rsplit('.', 1)[0] + '.html'
        fig.write_html(html_path)
        print(f"Interactive plot saved to: {html_path}")

    return fig


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics table."""
    base_row = df[df['exp_type'] == 'base'].iloc[0]
    base_prob = base_row['prob']
    base_log_prob = base_row['log_prob']

    summary_rows = []

    for exp_type in ['animal (filtered-v2)', 'control (vanilla)']:
        exp_data = df[df['exp_type'] == exp_type].sort_values('epoch')

        # Get epoch 0 and final epoch
        epoch_0 = exp_data[exp_data['epoch'] == 0].iloc[0]
        epoch_final = exp_data[exp_data['epoch'] == exp_data['epoch'].max()].iloc[0]

        summary_rows.append({
            'Model': exp_type,
            'Base Prob': base_prob,
            'Epoch 0 Prob': epoch_0['prob'],
            'Final Prob': epoch_final['prob'],
            'Change vs Base': epoch_final['prob'] - base_prob,
            'Change %': ((epoch_final['prob'] / base_prob) - 1) * 100,
            'Log Prob Change': epoch_final['log_prob'] - base_log_prob,
        })

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description='Plot probability comparison across epochs')
    parser.add_argument('--filtered-json', type=str, required=True,
                        help='Path to filtered-v2 results JSON')
    parser.add_argument('--vanilla-json', type=str, required=True,
                        help='Path to vanilla results JSON')
    parser.add_argument('--output', type=str, default='probability_comparison_plot.png',
                        help='Output path for the plot')
    parser.add_argument('--simple', action='store_true',
                        help='Create simple single-panel plot')

    args = parser.parse_args()

    # Create combined DataFrame
    df = create_comparison_df(args.filtered_json, args.vanilla_json)

    print("\nData Summary:")
    print(df.to_string(index=False))

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary = create_summary_table(df)
    print(summary.to_string(index=False))

    # Create plot
    if args.simple:
        plot_simple_comparison(df, args.output)
    else:
        plot_comparison(df, args.output)


if __name__ == '__main__':
    main()
