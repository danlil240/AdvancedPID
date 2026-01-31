"""
Visualization tools for system identification and autotuning results.
"""

from typing import Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pid_control.identification.csv_reader import ExperimentalData
from pid_control.identification.system_identifier import IdentificationResult
from pid_control.identification.autotune_from_data import AutotuneFromDataResult


class IdentificationVisualizer:
    """Visualize system identification and autotuning results."""
    
    @staticmethod
    def plot_identification_result(
        result: IdentificationResult,
        actual_output: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot system identification results with model fit.
        
        Args:
            result: Identification result
            actual_output: Actual measured output
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        ax_response = fig.add_subplot(gs[0, :])
        ax_error = fig.add_subplot(gs[1, :])
        ax_params = fig.add_subplot(gs[2, 0])
        ax_gains = fig.add_subplot(gs[2, 1])
        
        t = result.time
        y_actual = actual_output
        y_sim = result.simulated_output
        
        ax_response.plot(t, y_actual, 'b-', linewidth=2, label='Actual Output', alpha=0.7)
        ax_response.plot(t, y_sim, 'r--', linewidth=2, label='Model Prediction')
        ax_response.grid(True, alpha=0.3)
        ax_response.set_xlabel('Time (s)', fontsize=11)
        ax_response.set_ylabel('Output', fontsize=11)
        ax_response.set_title(
            f'System Identification: {result.model.model_type} Model (R² = {result.fit_quality:.4f})',
            fontsize=13, fontweight='bold'
        )
        ax_response.legend(fontsize=10)
        
        error = y_actual - y_sim
        ax_error.plot(t, error, 'g-', linewidth=1.5)
        ax_error.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_error.grid(True, alpha=0.3)
        ax_error.set_xlabel('Time (s)', fontsize=11)
        ax_error.set_ylabel('Prediction Error', fontsize=11)
        ax_error.set_title('Model Prediction Error', fontsize=12)
        
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        ax_error.text(
            0.02, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
            transform=ax_error.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )
        
        param_names = ['K (Gain)', 'τ (Time Const)', 'θ (Dead Time)']
        param_values = [result.model.K, result.model.tau, result.model.theta]
        if result.model.tau2 is not None:
            param_names.append('τ₂')
            param_values.append(result.model.tau2)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax_params.bar(param_names, param_values, color=colors[:len(param_names)], alpha=0.7)
        ax_params.set_ylabel('Value', fontsize=11)
        ax_params.set_title('Transfer Function Parameters', fontsize=12)
        ax_params.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, param_values):
            height = bar.get_height()
            ax_params.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9
            )
        
        gain_names = ['Kp', 'Ki', 'Kd']
        gain_values = [
            result.recommended_gains['kp'],
            result.recommended_gains['ki'],
            result.recommended_gains['kd']
        ]
        
        bars = ax_gains.bar(gain_names, gain_values, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.7)
        ax_gains.set_ylabel('Gain Value', fontsize=11)
        ax_gains.set_title(f'Recommended PID Gains\n({result.tuning_rule})', fontsize=12)
        ax_gains.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, gain_values):
            height = bar.get_height()
            ax_gains.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_autotune_comparison(
        result: AutotuneFromDataResult,
        data: ExperimentalData,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of initial vs optimized PID gains.
        
        Args:
            result: Autotuning result
            data: Original experimental data
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1.5, 1])
        
        ax_data = fig.add_subplot(gs[0, :])
        ax_gains = fig.add_subplot(gs[1, 0])
        ax_cost = fig.add_subplot(gs[1, 1])
        ax_model = fig.add_subplot(gs[2, 0])
        ax_improvement = fig.add_subplot(gs[2, 1])
        
        ax_data.plot(data.time, data.output, 'b-', linewidth=2, label='Measured Output', alpha=0.7)
        ax_data.plot(data.time, data.input, 'r-', linewidth=1.5, label='Control Input', alpha=0.6)
        if data.setpoint is not None:
            ax_data.plot(data.time, data.setpoint, 'g--', linewidth=1.5, label='Setpoint', alpha=0.6)
        ax_data.grid(True, alpha=0.3)
        ax_data.set_xlabel('Time (s)', fontsize=11)
        ax_data.set_ylabel('Value', fontsize=11)
        ax_data.set_title('Experimental Data from CSV', fontsize=13, fontweight='bold')
        ax_data.legend(fontsize=10)
        
        gain_names = ['Kp', 'Ki', 'Kd']
        initial = [result.initial_gains[k] for k in ['kp', 'ki', 'kd']]
        optimized = [result.optimized_gains[k] for k in ['kp', 'ki', 'kd']]
        
        x = np.arange(len(gain_names))
        width = 0.35
        
        bars1 = ax_gains.bar(x - width/2, initial, width, label='Initial (Analytical)', 
                            color='#3498db', alpha=0.7)
        bars2 = ax_gains.bar(x + width/2, optimized, width, label='Optimized (Numerical)',
                            color='#e74c3c', alpha=0.7)
        
        ax_gains.set_ylabel('Gain Value', fontsize=11)
        ax_gains.set_title('PID Gains Comparison', fontsize=12, fontweight='bold')
        ax_gains.set_xticks(x)
        ax_gains.set_xticklabels(gain_names)
        ax_gains.legend(fontsize=9)
        ax_gains.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_gains.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8
                )
        
        if len(result.tuning_result.history) > 0:
            history = result.tuning_result.history
            costs = [h['cost'] for h in history]
            iterations = range(len(costs))
            
            ax_cost.plot(iterations, costs, 'b-', linewidth=2, marker='o', markersize=4)
            ax_cost.set_xlabel('Iteration', fontsize=11)
            ax_cost.set_ylabel('Cost Function Value', fontsize=11)
            ax_cost.set_title('Optimization Convergence', fontsize=12, fontweight='bold')
            ax_cost.grid(True, alpha=0.3)
            ax_cost.set_yscale('log')
            
            final_cost = result.tuning_result.cost
            ax_cost.axhline(y=final_cost, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_cost:.2f}')
            ax_cost.legend(fontsize=9)
        
        model = result.identification.model
        param_text = (
            f"Model Type: {model.model_type}\n"
            f"Gain (K): {model.K:.4f}\n"
            f"Time Constant (τ): {model.tau:.4f} s\n"
            f"Dead Time (θ): {model.theta:.4f} s\n"
        )
        if model.tau2 is not None:
            param_text += f"τ₂: {model.tau2:.4f} s\n"
        param_text += f"\nFit Quality (R²): {result.identification.fit_quality:.4f}"
        
        ax_model.text(
            0.5, 0.5, param_text,
            transform=ax_model.transAxes,
            fontsize=11,
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
        )
        ax_model.axis('off')
        ax_model.set_title('Identified System Model', fontsize=12, fontweight='bold')
        
        improvement_text = (
            f"Performance Improvement:\n\n"
            f"{result.improvement:.2f}%\n\n"
            f"Tuning Method:\n{result.identification.tuning_rule}\n\n"
            f"Optimizer:\n{result.tuning_result.message}"
        )
        
        color = '#2ecc71' if result.improvement > 0 else '#e74c3c'
        ax_improvement.text(
            0.5, 0.5, improvement_text,
            transform=ax_improvement.transAxes,
            fontsize=11,
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2)
        )
        ax_improvement.axis('off')
        ax_improvement.set_title('Optimization Summary', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_tuning_rules_comparison(
        rules_comparison: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of different tuning rules.
        
        Args:
            rules_comparison: Dictionary mapping rule names to PID gains
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        rules = list(rules_comparison.keys())
        kp_values = [rules_comparison[r]['kp'] for r in rules]
        ki_values = [rules_comparison[r]['ki'] for r in rules]
        kd_values = [rules_comparison[r]['kd'] for r in rules]
        
        x = np.arange(len(rules))
        
        axes[0].bar(x, kp_values, color='#e74c3c', alpha=0.7)
        axes[0].set_ylabel('Kp Value', fontsize=11)
        axes[0].set_title('Proportional Gain', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(rules, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(x, ki_values, color='#3498db', alpha=0.7)
        axes[1].set_ylabel('Ki Value', fontsize=11)
        axes[1].set_title('Integral Gain', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(rules, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].bar(x, kd_values, color='#2ecc71', alpha=0.7)
        axes[2].set_ylabel('Kd Value', fontsize=11)
        axes[2].set_title('Derivative Gain', fontsize=12, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(rules, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
