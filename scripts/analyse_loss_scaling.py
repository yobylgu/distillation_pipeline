#!/usr/bin/env python3
"""
Loss Scaling Analysis Tool for Knowledge Distillation Pipeline

This script analyzes loss component magnitudes to determine optimal scaling factors
for balanced training. It computes running statistics and provides scaling recommendations.

Usage:
    python scripts/analyse_loss_scaling.py --log_file results/run/training_metrics.csv
    python scripts/analyse_loss_scaling.py --training_dir results/run/
    python scripts/analyse_loss_scaling.py --live_analysis  # for real-time monitoring
"""

import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LossScalingAnalyzer:
    """Analyzes loss component magnitudes and provides scaling recommendations."""
    
    def __init__(self, target_ratio_range: Tuple[float, float] = (0.5, 2.0)):
        """
        Initialize the analyzer.
        
        Args:
            target_ratio_range: Target range for loss component ratios (min, max)
        """
        self.target_ratio_range = target_ratio_range
        self.components = ['ce', 'kl', 'pans', 'ast', 'focal', 'jsd', 'semantic']
        self.analysis_results = {}
        
    def load_training_metrics(self, file_path: str) -> pd.DataFrame:
        """Load training metrics from CSV file."""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} training steps from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return pd.DataFrame()
    
    def load_from_training_dir(self, training_dir: str) -> pd.DataFrame:
        """Load metrics from a training directory structure."""
        training_path = Path(training_dir)
        
        # Try multiple possible metric file locations
        possible_files = [
            training_path / "training_metrics.csv",
            training_path / "step_metrics.csv",  # New step-level metrics
            training_path / "metrics_final.csv"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return self.load_training_metrics(str(file_path))
        
        print(f"No metric files found in {training_dir}")
        return pd.DataFrame()
    
    def extract_loss_components(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract loss component arrays from training data."""
        component_data = {}
        
        for component in self.components:
            # Try different column naming conventions
            possible_names = [
                f'loss_{component}',  # Standard naming
                f'{component}_loss',  # Alternative naming
                f'train_loss_{component}',  # Detailed naming
                component  # Direct component name
            ]
            
            for col_name in possible_names:
                if col_name in df.columns:
                    values = df[col_name].dropna().values
                    if len(values) > 0:
                        component_data[component] = values
                        print(f"Found {len(values)} values for {component}")
                        break
        
        return component_data
    
    def compute_running_statistics(self, component_data: Dict[str, np.ndarray], 
                                 window_size: int = 50) -> Dict[str, Dict]:
        """Compute running mean and std for loss components."""
        stats = {}
        
        for component, values in component_data.items():
            if len(values) < window_size:
                # Use all available data if less than window size
                running_mean = np.mean(values)
                running_std = np.std(values)
                running_values = values
            else:
                # Compute rolling statistics
                rolling_mean = pd.Series(values).rolling(window=window_size).mean()
                rolling_std = pd.Series(values).rolling(window=window_size).std()
                
                # Take the last 50% of training for stability analysis
                start_idx = len(values) // 2
                running_mean = rolling_mean[start_idx:].mean()
                running_std = rolling_std[start_idx:].mean()
                running_values = values[start_idx:]
            
            stats[component] = {
                'mean': running_mean,
                'std': running_std,
                'min': np.min(values),
                'max': np.max(values),
                'recent_mean': np.mean(running_values),
                'recent_std': np.std(running_values),
                'count': len(values),
                'cv': running_std / running_mean if running_mean > 0 else float('inf')  # Coefficient of variation
            }
            
        return stats
    
    def analyze_component_ratios(self, stats: Dict[str, Dict]) -> Dict[str, Dict]:
        """Analyze ratios between loss components."""
        ratios = {}
        component_means = {comp: stats[comp]['recent_mean'] for comp in stats.keys()}
        
        for comp1 in component_means:
            ratios[comp1] = {}
            for comp2 in component_means:
                if comp1 != comp2 and component_means[comp2] > 0:
                    ratio = component_means[comp1] / component_means[comp2]
                    ratios[comp1][comp2] = ratio
        
        return ratios
    
    def generate_scaling_recommendations(self, stats: Dict[str, Dict], 
                                       ratios: Dict[str, Dict]) -> Dict[str, float]:
        """Generate scaling factor recommendations for balanced training."""
        recommendations = {}
        
        # Find the component with the smallest mean as baseline
        component_means = {comp: stats[comp]['recent_mean'] for comp in stats.keys()}
        if not component_means:
            return recommendations
            
        baseline_component = min(component_means, key=component_means.get)
        baseline_value = component_means[baseline_component]
        
        print(f"\nUsing {baseline_component} as baseline (mean={baseline_value:.6f})")
        
        for component, mean_value in component_means.items():
            if component == baseline_component:
                recommendations[component] = 1.0
            else:
                # Calculate scaling factor to bring component to baseline range
                current_ratio = mean_value / baseline_value
                
                if current_ratio < self.target_ratio_range[0]:
                    # Component too small, scale up
                    scale_factor = self.target_ratio_range[0] / current_ratio
                    recommendations[component] = scale_factor
                elif current_ratio > self.target_ratio_range[1]:
                    # Component too large, scale down
                    scale_factor = self.target_ratio_range[1] / current_ratio
                    recommendations[component] = scale_factor
                else:
                    # Component within target range
                    recommendations[component] = 1.0
        
        return recommendations
    
    def check_semantic_scaling_need(self, stats: Dict[str, Dict]) -> Dict[str, float]:
        """Specific analysis for semantic loss scaling (β parameter)."""
        semantic_analysis = {}
        
        if 'semantic' not in stats:
            print("Semantic loss component not found in training data")
            return semantic_analysis
        
        semantic_stats = stats['semantic']
        
        # Compare semantic loss magnitude to other components
        other_components = [comp for comp in stats.keys() if comp != 'semantic']
        if not other_components:
            return semantic_analysis
        
        other_means = [stats[comp]['recent_mean'] for comp in other_components]
        avg_other_mean = np.mean(other_means)
        
        semantic_mean = semantic_stats['recent_mean']
        ratio = semantic_mean / avg_other_mean if avg_other_mean > 0 else 0
        
        # Recommend semantic_loss_scale (β) parameter
        if ratio < 0.1:
            recommended_beta = 10.0  # Scale up significantly
        elif ratio < 0.5:
            recommended_beta = 5.0   # Default scaling
        elif ratio < 1.0:
            recommended_beta = 2.0   # Moderate scaling
        else:
            recommended_beta = 1.0   # No scaling needed
        
        semantic_analysis = {
            'current_ratio': ratio,
            'recommended_beta': recommended_beta,
            'semantic_mean': semantic_mean,
            'other_components_avg': avg_other_mean,
            'confidence': 'high' if semantic_stats['count'] > 100 else 'low'
        }
        
        return semantic_analysis
    
    def visualize_loss_trends(self, component_data: Dict[str, np.ndarray], 
                            output_dir: str = "loss_analysis_plots"):
        """Create visualization plots for loss component analysis."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot 1: Loss component trends over time
        plt.figure(figsize=(12, 8))
        for component, values in component_data.items():
            steps = np.arange(len(values))
            plt.plot(steps, values, label=f'{component} loss', alpha=0.7)
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss Value')
        plt.title('Loss Component Trends During Training')
        plt.legend()
        plt.yscale('log')  # Log scale for better visualization
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_trends.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Loss component distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (component, values) in enumerate(component_data.items()):
            if i < len(axes):
                axes[i].hist(values, bins=50, alpha=0.7, density=True)
                axes[i].set_title(f'{component} Loss Distribution')
                axes[i].set_xlabel('Loss Value')
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(component_data), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}/")
    
    def generate_report(self, stats: Dict[str, Dict], ratios: Dict[str, Dict],
                       recommendations: Dict[str, float], 
                       semantic_analysis: Dict[str, float]) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("LOSS SCALING ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Component Statistics
        report.append("\n📊 COMPONENT STATISTICS:")
        report.append("-" * 50)
        for component, stat in stats.items():
            report.append(f"{component.upper():<12} | Mean: {stat['mean']:.6f} | Std: {stat['std']:.6f} | CV: {stat['cv']:.3f}")
        
        # Component Ratios
        report.append("\n⚖️  COMPONENT RATIOS:")
        report.append("-" * 50)
        if ratios:
            base_comp = list(ratios.keys())[0]
            for comp2 in ratios[base_comp]:
                ratio_val = ratios[base_comp][comp2]
                status = "✅ BALANCED" if self.target_ratio_range[0] <= ratio_val <= self.target_ratio_range[1] else "❌ IMBALANCED"
                report.append(f"{base_comp}/{comp2:<12} | Ratio: {ratio_val:.3f} | {status}")
        
        # Scaling Recommendations
        report.append("\n🎯 SCALING RECOMMENDATIONS:")
        report.append("-" * 50)
        for component, scale in recommendations.items():
            if scale != 1.0:
                action = "SCALE UP" if scale > 1.0 else "SCALE DOWN"
                report.append(f"{component.upper():<12} | Factor: {scale:.3f} | Action: {action}")
            else:
                report.append(f"{component.upper():<12} | Factor: {scale:.3f} | Action: NO CHANGE")
        
        # Semantic Loss Analysis
        if semantic_analysis:
            report.append("\n🧠 SEMANTIC LOSS ANALYSIS:")
            report.append("-" * 50)
            report.append(f"Current semantic/others ratio: {semantic_analysis['current_ratio']:.3f}")
            report.append(f"Recommended β (semantic_loss_scale): {semantic_analysis['recommended_beta']:.1f}")
            report.append(f"Confidence level: {semantic_analysis['confidence']}")
        
        # Configuration Recommendations
        report.append("\n⚙️  CONFIGURATION RECOMMENDATIONS:")
        report.append("-" * 50)
        if 'semantic' in recommendations:
            beta = semantic_analysis.get('recommended_beta', 5.0)
            report.append(f"Add to config/defaults.py:")
            report.append(f"DEFAULT_SEMANTIC_LOSS_SCALE = {beta:.1f}")
            report.append(f"")
            report.append(f"Command line usage:")
            report.append(f"--semantic_loss_scale {beta:.1f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def run_analysis(self, input_path: str, output_dir: str = "loss_analysis_output") -> Dict:
        """Run complete loss scaling analysis."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load data
        if Path(input_path).is_file():
            df = self.load_training_metrics(input_path)
        else:
            df = self.load_from_training_dir(input_path)
        
        if df.empty:
            print("No training data found. Cannot perform analysis.")
            return {}
        
        # Extract loss components
        component_data = self.extract_loss_components(df)
        if not component_data:
            print("No loss components found in data.")
            return {}
        
        # Compute statistics
        stats = self.compute_running_statistics(component_data)
        ratios = self.analyze_component_ratios(stats)
        recommendations = self.generate_scaling_recommendations(stats, ratios)
        semantic_analysis = self.check_semantic_scaling_need(stats)
        
        # Generate visualizations
        self.visualize_loss_trends(component_data, f"{output_dir}/plots")
        
        # Generate report
        report = self.generate_report(stats, ratios, recommendations, semantic_analysis)
        
        # Save results
        results = {
            'statistics': stats,
            'ratios': ratios,
            'recommendations': recommendations,
            'semantic_analysis': semantic_analysis
        }
        
        with open(f"{output_dir}/analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(f"{output_dir}/analysis_report.txt", 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nFull results saved to {output_dir}/")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Analyze loss component scaling for knowledge distillation")
    parser.add_argument('--log_file', type=str, help="Path to training metrics CSV file")
    parser.add_argument('--training_dir', type=str, help="Path to training directory")
    parser.add_argument('--output_dir', type=str, default="loss_analysis_output", 
                       help="Output directory for analysis results")
    parser.add_argument('--target_min_ratio', type=float, default=0.5, 
                       help="Minimum target ratio between loss components")
    parser.add_argument('--target_max_ratio', type=float, default=2.0,
                       help="Maximum target ratio between loss components")
    parser.add_argument('--window_size', type=int, default=50,
                       help="Window size for running statistics")
    
    args = parser.parse_args()
    
    if not args.log_file and not args.training_dir:
        parser.error("Must specify either --log_file or --training_dir")
    
    # Initialize analyzer
    analyzer = LossScalingAnalyzer(target_ratio_range=(args.target_min_ratio, args.target_max_ratio))
    
    # Run analysis
    input_path = args.log_file if args.log_file else args.training_dir
    results = analyzer.run_analysis(input_path, args.output_dir)
    
    if results:
        print("\n✅ Analysis completed successfully!")
        
        # Print key recommendations
        if 'semantic_analysis' in results and results['semantic_analysis']:
            beta = results['semantic_analysis'].get('recommended_beta', 5.0)
            print(f"\n🎯 KEY RECOMMENDATION: Set semantic_loss_scale = {beta:.1f}")
    else:
        print("❌ Analysis failed - check input data")

if __name__ == "__main__":
    main()