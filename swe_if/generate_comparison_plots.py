# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any

class ComparativeVisualizer:
    """
    A class to generate comparative visualizations for code benchmarks.
    """
    def __init__(self):
        self.benchmarks = ['bigcodebench', 'livecodebench']
        self.colors = {
            'Single-Turn Generation': '#FF6B6B',
            'Multi-Turn Editing': '#4ECDC4'
        }
        
        # Set font and style configurations
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 22
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['legend.fontsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

    def _load_and_process_data(self, benchmark: str) -> Dict[str, np.ndarray]:
        """Loads and processes data for a single benchmark."""
        print(f"Processing data for {benchmark}...")
        
        # --- Functional Data ---
        func_file = f"scores/{benchmark}/func_results.csv"
        if not os.path.exists(func_file): raise FileNotFoundError(f"File not found: {func_file}")
        df_func = pd.read_csv(func_file, header=None)
        gen_avg_func = np.mean(df_func.iloc[2:, 1:7].astype(float).values, axis=0)
        edit_avg_func = np.mean(df_func.iloc[2:, 7:13].astype(float).values, axis=0)
        baseline = gen_avg_func[0]
        gen_regression = (baseline - gen_avg_func) / baseline * 100
        edit_regression = (baseline - edit_avg_func) / baseline * 100

        # --- Task-level IF Data ---
        if_file = f"scores/{benchmark}/if_results.csv"
        if not os.path.exists(if_file): raise FileNotFoundError(f"File not found: {if_file}")
        df_if = pd.read_csv(if_file, header=None)
        task_gen_avg_if = np.mean(df_if.iloc[3:, 6:11].astype(float).values, axis=0)
        task_edit_avg_if = np.mean(df_if.iloc[3:, 16:21].astype(float).values, axis=0)
        
        # --- Position Analysis Data ---
        pos_file = f"scores/{benchmark}/position_results.csv"
        if not os.path.exists(pos_file): raise FileNotFoundError(f"File not found: {pos_file}")
        df_pos = pd.read_csv(pos_file, header=None)
        pos_gen_avg_if = np.mean(df_pos.iloc[2:, 1:6].astype(float).values, axis=0)
        pos_edit_avg_if = np.mean(df_pos.iloc[2:, 6:11].astype(float).values, axis=0)
        
        return {
            "gen_regression": gen_regression,
            "edit_regression": edit_regression,
            "task_gen_if": task_gen_avg_if,
            "task_edit_if": task_edit_avg_if,
            "position_gen_if": pos_gen_avg_if,
            "position_edit_if": pos_edit_avg_if,
        }

    def plot_functional_comparison(self, all_data: Dict[str, Any]):
        """Plots functional regression. (Size: 12x6)"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        bench_titles = {'bigcodebench': 'Big-SWE-IF', 'livecodebench': 'Live-SWE-IF'}
        
        for i, benchmark in enumerate(self.benchmarks):
            ax = axes[i]
            data = all_data[benchmark]
            instructions = np.arange(6)
            ax.plot(instructions, data['gen_regression'], 'o-', 
                    color=self.colors['Single-Turn Generation'], linewidth=3, markersize=8, 
                    label='Single-Turn Generation')
            ax.plot(instructions, data['edit_regression'], 'o-', 
                    color=self.colors['Multi-Turn Editing'], linewidth=3, markersize=8, 
                    label='Multi-Turn Editing')
            ax.set_title(bench_titles[benchmark])
            ax.set_xlabel('Number of Instructions')
            ax.set_xticks(instructions)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            
        axes[0].set_ylabel('Functional Regression (%)')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.1))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        save_path = "figures/functional_comparison.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Functional comparison plot saved to: {save_path}")
        plt.close(fig)

    def plot_task_level_if_comparison(self, all_data: Dict[str, Any]):
        """Plots task-level IF scores. (Size: 12x6)"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        bench_titles = {'bigcodebench': 'Big-SWE-IF', 'livecodebench': 'Live-SWE-IF'}

        for i, benchmark in enumerate(self.benchmarks):
            ax = axes[i]
            data = all_data[benchmark]
            instructions = np.arange(1, 6)
            ax.plot(instructions, data['task_gen_if'], 'o-', 
                    color=self.colors['Single-Turn Generation'], linewidth=3, markersize=8, 
                    label='Single-Turn Generation')
            ax.plot(instructions, data['task_edit_if'], 'o-', 
                    color=self.colors['Multi-Turn Editing'], linewidth=3, markersize=8, 
                    label='Multi-Turn Editing')
            ax.set_title(bench_titles[benchmark])
            ax.set_xlabel('Number of Instructions')
            ax.set_xticks(instructions)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        axes[0].set_ylabel('Task-level IF Score')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.1))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        save_path = "figures/task_level_if_comparison.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Task-level IF comparison plot saved to: {save_path}")
        plt.close(fig)

    def plot_position_if_comparison(self, all_data: Dict[str, Any]):
        """
        Plots instruction position analysis with the same aspect ratio as other plots.
        """
        # MODIFICATION: Reverted figsize to 12x6 to match other plots.
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
        bench_titles = {'bigcodebench': 'Big-SWE-IF', 'livecodebench': 'Live-SWE-IF'}

        for i, benchmark in enumerate(self.benchmarks):
            ax = axes[i]
            data = all_data[benchmark]
            positions = np.arange(1, 6)
            ax.plot(positions, data['position_gen_if'], 'o-', 
                    color=self.colors['Single-Turn Generation'], linewidth=3, markersize=8, 
                    label='Single-Turn Generation')
            ax.plot(positions, data['position_edit_if'], 'o-', 
                    color=self.colors['Multi-Turn Editing'], linewidth=3, markersize=8, 
                    label='Multi-Turn Editing')
            ax.set_title(bench_titles[benchmark])
            ax.set_xlabel('Instruction Position')
            ax.set_xticks(positions)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        axes[0].set_ylabel('Instruction-level IF')
        handles, labels = axes[0].get_legend_handles_labels()
        
        # MODIFICATION: Reverted legend position to match other plots.
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.1))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        
        save_path = "figures/position_if_comparison.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position analysis comparison plot saved to: {save_path}")
        plt.close(fig)

    def run(self):
        """Main execution method to generate all plots."""
        try:
            all_benchmark_data = {
                benchmark: self._load_and_process_data(benchmark) 
                for benchmark in self.benchmarks
            }
            
            self.plot_functional_comparison(all_benchmark_data)
            self.plot_task_level_if_comparison(all_benchmark_data)
            self.plot_position_if_comparison(all_benchmark_data)
            
            print("\nSuccessfully generated all three comparison plots!")
            
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Please ensure the required CSV files are in their respective directories.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

def main():
    visualizer = ComparativeVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()