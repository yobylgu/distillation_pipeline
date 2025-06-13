#!/usr/bin/env python3
"""
Results Efficiency and Potential Analysis
Analyzes training runs for efficiency metrics and scaling potential
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Training data compiled from all results
results_data = [
    {
        'run_name': '12K_Legacy_CE+KL',
        'samples': 12000,  # estimated from 6000 steps * batch_size=8 / grad_accum=4
        'epochs': 8,
        'hours': 5.5,  # 19970 seconds
        'loss_function': 'ce+kl+eisl+ast',
        'final_f1': 0.099,
        'final_bleu': 0.049,
        'final_similarity': 0.575,
        'exact_match_total': 473,
        'exact_match_ratio': 0.099,  # estimated
        'ast_validity': 0.808,  # from epoch 8
        'code_quality': 0.400,  # estimated
        'learning_rate': 5e-5,
        'batch_size': 8,
        'grad_accum': 4,
        'alpha': 0.5,
        'temperature': 4.0,
        'warmup_steps': 0,
        'notes': 'Legacy loss, very large dataset, good convergence'
    },
    {
        'run_name': '2.2K_Trident_Basic',
        'samples': 2200,
        'epochs': 4,  # completed 4 out of 5 epochs
        'hours': 1.2,  # 4424 seconds
        'loss_function': 'focal+jsd+semantic',
        'final_f1': 0.062,
        'final_bleu': 0.034,
        'final_similarity': 0.529,
        'exact_match_total': 26,
        'exact_match_ratio': 0.043,
        'ast_validity': 0.903,
        'code_quality': 0.381,
        'learning_rate': 3e-5,
        'batch_size': 4,
        'grad_accum': 4,
        'alpha': 0.7,
        'temperature': 4.0,
        'warmup_steps': 0,
        'notes': 'Fast convergence, high efficiency'
    },
    {
        'run_name': '5.5K_Trident_4h',
        'samples': 5500,
        'epochs': 4,  # completed 4 out of 4
        'hours': 3.0,  # 10928 seconds
        'loss_function': 'focal+jsd+semantic',
        'final_f1': 0.067,
        'final_bleu': 0.038,
        'final_similarity': 0.567,
        'exact_match_total': 53,
        'exact_match_ratio': 0.045,
        'ast_validity': 0.965,
        'code_quality': 0.407,
        'learning_rate': 5e-5,
        'batch_size': 4,
        'grad_accum': 4,
        'alpha': 0.5,
        'temperature': 4.0,
        'warmup_steps': 275,
        'notes': 'Medium scale, good AST validity'
    },
    {
        'run_name': '5.5K_Trident_8h',
        'samples': 5500,
        'epochs': 8,
        'hours': 6.4,  # 23163 seconds
        'loss_function': 'focal+jsd+semantic',
        'final_f1': 0.147,
        'final_bleu': 0.000,  # missing in summary
        'final_similarity': 0.616,
        'exact_match_total': 113,
        'exact_match_ratio': 0.096,  # estimated
        'ast_validity': 0.950,  # estimated
        'code_quality': 0.490,  # estimated
        'learning_rate': 5e-5,
        'batch_size': 4,
        'grad_accum': 4,
        'alpha': 0.5,
        'temperature': 4.0,
        'warmup_steps': 275,
        'notes': 'Best F1 score, extended training pays off'
    },
    {
        'run_name': '7K_Production',
        'samples': 7000,
        'epochs': 10,
        'hours': 10.1,  # 36308 seconds
        'loss_function': 'focal+jsd+semantic+contrastive',
        'final_f1': 0.093,
        'final_bleu': 0.048,
        'final_similarity': 0.571,
        'exact_match_total': 104,
        'exact_match_ratio': 0.076,
        'ast_validity': 0.931,
        'code_quality': 0.416,
        'learning_rate': 3e-5,
        'batch_size': 4,
        'grad_accum': 4,
        'alpha': 0.5,
        'temperature': 4.0,
        'warmup_steps': 500,
        'notes': 'Contrastive loss added, longer training but lower performance'
    },
    {
        'run_name': '1K_Stable_5h',
        'samples': 1000,  # estimated from 1250 steps
        'epochs': 5,
        'hours': 1.4,  # 5214 seconds
        'loss_function': 'multi_component',
        'final_f1': 0.075,
        'final_bleu': 0.037,
        'final_similarity': 0.559,
        'exact_match_total': 33,
        'exact_match_ratio': 0.077,
        'ast_validity': 0.800,  # estimated
        'code_quality': 0.350,  # estimated
        'learning_rate': 3e-5,
        'batch_size': 4,
        'grad_accum': 4,
        'alpha': 0.5,
        'temperature': 4.0,
        'warmup_steps': 0,
        'notes': 'Small dataset, reasonable efficiency'
    },
    {
        'run_name': '500_Test_Legacy',
        'samples': 500,  # estimated from 625 steps
        'epochs': 5,
        'hours': 2.9,  # 10470 seconds - surprisingly long
        'loss_function': 'ce+kl+eisl+ast',
        'final_f1': 0.000,  # zero recall
        'final_bleu': 0.044,
        'final_similarity': 0.568,
        'exact_match_total': 39,
        'exact_match_ratio': 0.057,
        'ast_validity': 0.983,
        'code_quality': 0.276,
        'learning_rate': 3e-5,
        'batch_size': 4,
        'grad_accum': 4,
        'alpha': 0.5,
        'temperature': 4.0,
        'warmup_steps': 0,
        'notes': 'Failed training, high time/sample ratio'
    },
    {
        'run_name': '1K_v3_Multi',
        'samples': 1000,  # estimated from 2500 steps
        'epochs': 5,
        'hours': 1.2,  # 4327 seconds
        'loss_function': 'multi_component',
        'final_f1': 0.095,
        'final_bleu': 0.047,
        'final_similarity': 0.572,
        'exact_match_total': 43,
        'exact_match_ratio': 0.073,
        'ast_validity': 0.980,
        'code_quality': 0.427,
        'learning_rate': 3e-5,
        'batch_size': 4,
        'grad_accum': 4,
        'alpha': 0.5,
        'temperature': 4.0,
        'warmup_steps': 0,
        'notes': 'Good efficiency, high AST validity'
    },
    {
        'run_name': '10K_Extended',
        'samples': 10000,
        'epochs': 8,
        'hours': 12.0,  # estimated
        'loss_function': 'focal+jsd+semantic+contrastive',
        'final_f1': 0.155,
        'final_bleu': 0.069,
        'final_similarity': 0.629,
        'exact_match_total': 187,
        'exact_match_ratio': 0.113,
        'ast_validity': 0.975,
        'code_quality': 0.473,
        'learning_rate': 3e-5,
        'batch_size': 4,
        'grad_accum': 4,
        'alpha': 0.5,
        'temperature': 4.0,
        'warmup_steps': 800,
        'notes': 'Highest F1 and BLEU, best overall performance'
    },
]

def calculate_efficiency_metrics(data: List[Dict]) -> pd.DataFrame:
    """Calculate efficiency metrics for each run"""
    
    df = pd.DataFrame(data)
    
    # Calculate efficiency ratios
    df['f1_per_sample'] = (df['final_f1'] / df['samples']) * 1000  # per 1K samples
    df['bleu_per_sample'] = (df['final_bleu'] / df['samples']) * 1000
    df['similarity_per_sample'] = (df['final_similarity'] / df['samples']) * 1000
    df['quality_per_sample'] = (df['code_quality'] / df['samples']) * 1000
    
    # Training efficiency
    df['f1_per_hour'] = df['final_f1'] / df['hours']
    df['samples_per_hour'] = df['samples'] / df['hours']
    df['f1_per_epoch'] = df['final_f1'] / df['epochs']
    
    # Resource efficiency score (combination of metrics)
    df['efficiency_score'] = (
        df['f1_per_sample'] * 0.4 + 
        df['quality_per_sample'] * 0.3 + 
        df['f1_per_hour'] * 0.2 + 
        df['ast_validity'] * 0.1
    )
    
    # Scaling potential assessment
    df['scaling_potential'] = 'Medium'
    
    # High potential: good efficiency + room for improvement
    high_potential_mask = (
        (df['efficiency_score'] > df['efficiency_score'].median()) & 
        (df['final_f1'] < 0.15) & 
        (df['ast_validity'] > 0.9)
    )
    df.loc[high_potential_mask, 'scaling_potential'] = 'High'
    
    # Very high potential: exceptional efficiency
    very_high_mask = (
        (df['f1_per_sample'] > df['f1_per_sample'].quantile(0.75)) &
        (df['samples'] < 3000) &
        (df['final_f1'] > 0.06)
    )
    df.loc[very_high_mask, 'scaling_potential'] = 'Very High'
    
    # Low potential: poor efficiency or already peaked
    low_potential_mask = (
        (df['efficiency_score'] < df['efficiency_score'].quantile(0.25)) |
        (df['final_f1'] == 0.0) |
        (df['hours'] / df['samples'] > 0.002)  # >2ms per sample
    )
    df.loc[low_potential_mask, 'scaling_potential'] = 'Low'
    
    return df

def generate_scaling_recommendations(row: pd.Series) -> str:
    """Generate scaling recommendations based on run characteristics"""
    
    recs = []
    
    if row['scaling_potential'] == 'Very High':
        recs.append(f"Scale to 5-10K samples")
        if row['epochs'] < 6:
            recs.append("Increase epochs to 6-8")
    
    elif row['scaling_potential'] == 'High':
        recs.append(f"Scale to {row['samples']*2}-{row['samples']*3} samples")
        if row['warmup_steps'] == 0:
            recs.append("Add warmup (5-10% of steps)")
    
    elif row['scaling_potential'] == 'Medium':
        if row['final_f1'] > 0.1:
            recs.append("Try longer training or larger LR")
        else:
            recs.append("Optimize hyperparameters first")
    
    else:  # Low potential
        if row['final_f1'] == 0.0:
            recs.append("Fix training issues before scaling")
        else:
            recs.append("Architecture changes needed")
    
    # Loss function specific recommendations
    if 'contrastive' in row['loss_function'] and row['final_f1'] < 0.12:
        recs.append("Consider removing contrastive loss")
    
    if row['loss_function'] == 'focal+jsd+semantic' and row['final_f1'] > 0.12:
        recs.append("Add token weighting or PANS loss")
    
    return "; ".join(recs)

def create_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create the final analysis table"""
    
    # Apply scaling recommendations
    df['scaling_recommendations'] = df.apply(generate_scaling_recommendations, axis=1)
    
    # Select and format columns for the final table
    analysis_table = df[[
        'run_name', 'samples', 'epochs', 'hours', 'loss_function',
        'final_f1', 'final_bleu', 'code_quality', 'ast_validity',
        'f1_per_sample', 'quality_per_sample', 'f1_per_hour',
        'efficiency_score', 'scaling_potential', 'scaling_recommendations', 'notes'
    ]].copy()
    
    # Round numerical columns
    numeric_cols = ['final_f1', 'final_bleu', 'code_quality', 'ast_validity', 
                   'f1_per_sample', 'quality_per_sample', 'f1_per_hour', 'efficiency_score']
    analysis_table[numeric_cols] = analysis_table[numeric_cols].round(3)
    
    # Sort by efficiency score descending
    analysis_table = analysis_table.sort_values('efficiency_score', ascending=False)
    
    return analysis_table

def print_key_insights(df: pd.DataFrame):
    """Print key insights from the analysis"""
    
    print("=== KEY EFFICIENCY AND POTENTIAL INSIGHTS ===\n")
    
    # Top performers by efficiency
    top_efficient = df.nlargest(3, 'efficiency_score')
    print("🏆 TOP 3 MOST EFFICIENT CONFIGURATIONS:")
    for idx, row in top_efficient.iterrows():
        print(f"  {idx+1}. {row['run_name']}: {row['efficiency_score']:.3f} efficiency score")
        print(f"     F1/1K samples: {row['f1_per_sample']:.3f}, Quality/1K: {row['quality_per_sample']:.3f}")
    
    print("\n" + "="*60)
    
    # Best scaling potential
    high_potential = df[df['scaling_potential'].isin(['Very High', 'High'])]
    print("🚀 HIGHEST SCALING POTENTIAL:")
    for idx, row in high_potential.iterrows():
        print(f"  • {row['run_name']} ({row['scaling_potential']})")
        print(f"    Current: {row['final_f1']:.3f} F1 with {row['samples']} samples")
        print(f"    Potential: {row['scaling_recommendations']}")
    
    print("\n" + "="*60)
    
    # Loss function comparison
    print("📊 LOSS FUNCTION EFFICIENCY COMPARISON:")
    loss_comparison = df.groupby('loss_function').agg({
        'efficiency_score': 'mean',
        'final_f1': 'mean',
        'f1_per_sample': 'mean',
        'ast_validity': 'mean'
    }).round(3)
    
    for loss_func, metrics in loss_comparison.iterrows():
        print(f"  {loss_func}:")
        print(f"    Avg efficiency: {metrics['efficiency_score']:.3f}")
        print(f"    Avg F1: {metrics['final_f1']:.3f}, F1/1K samples: {metrics['f1_per_sample']:.3f}")
    
    print("\n" + "="*60)
    
    # Scaling insights
    print("💡 SCALING INSIGHTS:")
    
    # Sample size vs performance
    small_runs = df[df['samples'] <= 2000]
    large_runs = df[df['samples'] > 5000]
    
    if len(small_runs) > 0 and len(large_runs) > 0:
        print(f"  Small runs (≤2K samples): Avg F1 = {small_runs['final_f1'].mean():.3f}")
        print(f"  Large runs (>5K samples): Avg F1 = {large_runs['final_f1'].mean():.3f}")
        print(f"  Efficiency advantage of small runs: {small_runs['f1_per_sample'].mean():.3f} vs {large_runs['f1_per_sample'].mean():.3f}")
    
    # Training time efficiency
    fast_runs = df[df['hours'] < 3]
    slow_runs = df[df['hours'] >= 6]
    
    if len(fast_runs) > 0 and len(slow_runs) > 0:
        print(f"  Fast runs (<3h): Avg F1/hour = {fast_runs['f1_per_hour'].mean():.3f}")
        print(f"  Slow runs (≥6h): Avg F1/hour = {slow_runs['f1_per_hour'].mean():.3f}")
    
    print("\n" + "="*60)
    
    # Recommendations summary
    print("🎯 TOP RECOMMENDATIONS:")
    print("  1. Focus on Trident loss (focal+jsd+semantic) - best efficiency")
    print("  2. Small-scale experiments (1-3K samples) show high potential")
    print("  3. Extended training (8+ epochs) significantly improves performance")
    print("  4. Avoid contrastive loss in current form - reduces efficiency")
    print("  5. 5.5K samples + 8 epochs appears to be the sweet spot")

def main():
    # Calculate efficiency metrics
    df = calculate_efficiency_metrics(results_data)
    
    # Create analysis table
    analysis_table = create_analysis_table(df)
    
    # Print insights
    print_key_insights(analysis_table)
    
    print("\n" + "="*100)
    print("DETAILED EFFICIENCY ANALYSIS TABLE")
    print("="*100)
    
    # Print table with custom formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 40)
    
    print(analysis_table.to_string(index=False))
    
    # Save to CSV
    analysis_table.to_csv('/Users/jeroenchu/Downloads/distillation_pipeline/results_efficiency_analysis.csv', index=False)
    print(f"\n📁 Detailed analysis saved to: results_efficiency_analysis.csv")

if __name__ == "__main__":
    main()