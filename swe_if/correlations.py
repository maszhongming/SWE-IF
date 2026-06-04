#!/usr/bin/env python3
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

"""
AI Model Scores Correlation Analysis
Calculate correlations between different benchmark weight combinations and Arena scores.
Supports BigCodeBench (BCB) and LiveCodeBench (LCB) with configurable instruction following.

This module merges two previously separate scripts via the ``--format`` flag:

  --format png : Pearson + Spearman + Kendall (3 metrics).
                 Produces individual per-correlation line plots plus a 3-panel
                 combined ``overall_correlations.png`` under ``figures/{benchmark}/``.
                 CSV (under ``correlation_results/{benchmark}/``) includes the
                 ``kendall`` column. This is the source of the paper's appendix figures.

  --format pdf : Pearson + Spearman (2 metrics).
                 Produces a single side-by-side 2-panel, extra-large-font PDF under
                 ``figures_pdf/{benchmark}/``. CSV (under
                 ``correlation_results/{benchmark}/``) has no ``kendall`` column.
                 This is the source of the paper's main-text Figure 5.

Last Updated: Sep 18, 2025
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Model scores data
# Last Updated: Sep 18, 2025
scores = {
    'gemini-3.1-pro-preview': {
        'bcb_score': {
            'func_score': 50.35,
            'if_score': {
                '1_inst': 82.19,
                '3_inst': 79.18,
                '5_inst': 79.47,
            }
        },
        'lcb_score': {
            'func_score': 85.31,
            'if_score': {
                '1_inst': 75.83,
                '3_inst': 76.21,
                '5_inst': 76.87,
            }
        },
        'arena_rating': {
            'default': 1470,
            'remove_style_control': 1468,
        }
    },
    'gemini-2.5-flash': {
        'bcb_score': {
            'func_score': 47.37,
            'if_score': {
                '1_inst': 81.67,
                '3_inst': 77.34,
                '5_inst': 75.91,
            }
        },
        'lcb_score': {
            'func_score': 74.50,
            'if_score': {
                '1_inst': 66.54,
                '3_inst': 67.84,
                '5_inst': 68.13,
            }
        },
        'arena_rating': {
            'default': 1419,
            'remove_style_control': 1422,
        }
    },
    'gemini-2.0-flash-lite': {
        'bcb_score': {
            'func_score': 46.93,
            'if_score': {
                '1_inst': 70.44,
                '3_inst': 69.30,
                '5_inst': 68.89,
            }
        },
        'lcb_score': {
            'func_score': 34.12,
            'if_score': {
                '1_inst': 62.94,
                '3_inst': 65.28,
                '5_inst': 65.69,
            }
        },
        'arena_rating': {
            'default': 1352,
            'remove_style_control': 1336,
        }
    },
    'claude-opus-4-20250514-thinking': {
        'bcb_score': {
            'func_score': 51.05,
            'if_score': {
                '1_inst': 88.77,
                '3_inst': 86.05,
                '5_inst': 85.60,
            }
        },
        'lcb_score': {
            'func_score': 68.72,
            'if_score': {
                '1_inst': 78.86,
                '3_inst': 77.54,
                '5_inst': 78.75,
            }
        },
        'arena_rating': {
            'default': 1481,
            'remove_style_control': 1430,
        }
    },
    'claude-sonnet-4-20250514-thinking': {
        'bcb_score': {
            'func_score': 51.84,
            'if_score': {
                '1_inst': 84.91,
                '3_inst': 81.37,
                '5_inst': 81.37,
            }
        },
        'lcb_score': {
            'func_score': 66.35,
            'if_score': {
                '1_inst': 75.73,
                '3_inst': 75.29,
                '5_inst': 75.20,
            }
        },
        'arena_rating': {
            'default': 1460,
            'remove_style_control': 1407,
        }
    },
    'claude-3-7-sonnet-20250219-thinking': {
        'bcb_score': {
            'func_score': 51.32,
            'if_score': {
                '1_inst': 80.26,
                '3_inst': 75.47,
                '5_inst': 74.26,
            }
        },
        'lcb_score': {
            'func_score': 61.80,
            'if_score': {
                '1_inst': 72.42,
                '3_inst': 68.18,
                '5_inst': 68.38,
            }
        },
        'arena_rating': {
            'default': 1430,
            'remove_style_control': 1353,
        }
    },
    'claude-3-5-sonnet-20241022': {
        'bcb_score': {
            'func_score': 48.42,
            'if_score': {
                '1_inst': 80.61,
                '3_inst': 76.02,
                '5_inst': 74.70,
            }
        },
        'lcb_score': {
            'func_score': 45.40,
            'if_score': {
                '1_inst': 70.52,
                '3_inst': 68.56,
                '5_inst': 67.28,
            }
        },
        'arena_rating': {
            'default': 1418,
            'remove_style_control': 1337,
        }
    },
    'claude-3-5-haiku-20241022': {
        'bcb_score': {
            'func_score': 46.58,
            'if_score': {
                '1_inst': 64.56,
                '3_inst': 63.71,
                '5_inst': 63.82,
            }
        },
        'lcb_score': {
            'func_score': 37.63,
            'if_score': {
                '1_inst': 63.22,
                '3_inst': 61.45,
                '5_inst': 62.77,
            }
        },
        'arena_rating': {
            'default': 1370,
            'remove_style_control': 1285,
        }
    },
    'claude-3-haiku-20240307': {
        'bcb_score': {
            'func_score': 38.07,
            'if_score': {
                '1_inst': 67.89,
                '3_inst': 64.88,
                '5_inst': 64.53,
            }
        },
        'lcb_score': {
            'func_score': 22.09,
            'if_score': {
                '1_inst': 61.61,
                '3_inst': 60.98,
                '5_inst': 60.45,
            }
        },
        'arena_rating': {
            'default': 1287,
            'remove_style_control': 1202,
        }
    },
    'deepseek-r1-0528': {
        'bcb_score': {
            'func_score': 49.21,
            'if_score': {
                '1_inst': 74.04,
                '3_inst': 69.01,
                '5_inst': 67.71,
            }
        },
        'lcb_score': {
            'func_score': None,
            'if_score': {
                '1_inst': None,
                '3_inst': None,
                '5_inst': None,
            }
        },
        'arena_rating': {
            'default': 1458,
            'remove_style_control': 1436,
        }
    },
    'deepseek-v3-0324': {
        'bcb_score': {
            'func_score': 50.18,
            'if_score': {
                '1_inst': 67.89,
                '3_inst': 64.04,
                '5_inst': 65.09,
            }
        },
        'lcb_score': {
            'func_score': 57.25,
            'if_score': {
                '1_inst': 52.80,
                '3_inst': 55.67,
                '5_inst': 55.79,
            }
        },
        'arena_rating': {
            'default': 1431,
            'remove_style_control': 1389,
        }
    },
    'gpt-5': {
        'bcb_score': {
            'func_score': 46.49,
            'if_score': {
                '1_inst': 82.89,
                '3_inst': 81.96,
                '5_inst': 81.77,
            }
        },
        'lcb_score': {
            'func_score': 71.47,
            'if_score': {
                '1_inst': 82.18,
                '3_inst': 81.86,
                '5_inst': 82.82,
            }
        },
        'arena_rating': {
            'default': 1467,
            'remove_style_control': 1440,
        }
    },
    'o4-mini-2025-04-16': {
        'bcb_score': {
            'func_score': 52.28,
            'if_score': {
                '1_inst': 84.82,
                '3_inst': 83.25,
                '5_inst': 84.25,
            }
        },
        'lcb_score': {
            'func_score': 80.95,
            'if_score': {
                '1_inst': 73.18,
                '3_inst': 73.33,
                '5_inst': 73.82,
            }
        },
        'arena_rating': {
            'default': 1428,
            'remove_style_control': 1380,
        }
    },
    'o3-mini-high': {
        'bcb_score': {
            'func_score': 49.91,
            'if_score': {
                '1_inst': 80.70,
                '3_inst': 73.63,
                '5_inst': 71.68,
            }
        },
        'lcb_score': {
            'func_score': None,
            'if_score': {
                '1_inst': None,
                '3_inst': None,
                '5_inst': None,
            }
        },
        'arena_rating': {
            'default': 1421,
            'remove_style_control': 1379,
        }
    },
    'gpt-4.1-2025-04-14': {
        'bcb_score': {
            'func_score': 47.54,
            'if_score': {
                '1_inst': 81.40,
                '3_inst': 78.60,
                '5_inst': 77.75,
            }
        },
        'lcb_score': {
            'func_score': 53.08,
            'if_score': {
                '1_inst': 68.63,
                '3_inst': 66.76,
                '5_inst': 66.67,
            }
        },
        'arena_rating': {
            'default': 1447,
            'remove_style_control': 1399,
        }
    },
    'gpt-4.1-mini-2025-04-14': {
        'bcb_score': {
            'func_score': 49.04,
            'if_score': {
                '1_inst': 78.16,
                '3_inst': 75.15,
                '5_inst': 73.42,
            }
        },
        'lcb_score': {
            'func_score': 58.86,
            'if_score': {
                '1_inst': 67.20,
                '3_inst': 68.63,
                '5_inst': 67.41,
            }
        },
        'arena_rating': {
            'default': 1423,
            'remove_style_control': 1371,
        }
    },
    'gpt-4o-2024-08-06': {
        'bcb_score': {
            'func_score': 49.82,
            'if_score': {
                '1_inst': 77.46,
                '3_inst': 74.56,
                '5_inst': 73.44,
            }
        },
        'lcb_score': {
            'func_score': 42.75,
            'if_score': {
                '1_inst': 60.85,
                '3_inst': 61.48,
                '5_inst': 61.93,
            }
        },
        'arena_rating': {
            'default': 1352,
            'remove_style_control': 1289,
        }
    },
    'gpt-4o-mini-2024-07-18': {
        'bcb_score': {
            'func_score': 46.05,
            'if_score': {
                '1_inst': 76.40,
                '3_inst': 73.13,
                '5_inst': 73.32,
            }
        },
        'lcb_score': {
            'func_score': 22.27,
            'if_score': {
                '1_inst': 65.88,
                '3_inst': 65.72,
                '5_inst': 65.63,
            }
        },
        'arena_rating': {
            'default': 1340,
            'remove_style_control': 1297,
        }
    },
    'grok-4-0709': {
        'bcb_score': {
            'func_score': 53.07,
            'if_score': {
                '1_inst': 87.11,
                '3_inst': 84.77,
                '5_inst': 84.81,
            }
        },
        'lcb_score': {
            'func_score': None,
            'if_score': {
                '1_inst': None,
                '3_inst': None,
                '5_inst': None,
            }
        },
        'arena_rating': {
            'default': 1440,
            'remove_style_control': 1431,
        }
    },
    'grok-3-mini-beta': {
        'bcb_score': {
            'func_score': 48.77,
            'if_score': {
                '1_inst': 82.81,
                '3_inst': 78.25,
                '5_inst': 77.46,
            }
        },
        'lcb_score': {
            'func_score': 65.97,
            'if_score': {
                '1_inst': 70.05,
                '3_inst': 69.61,
                '5_inst': 68.99,
            }
        },
        'arena_rating': {
            'default': 1384,
            'remove_style_control': 1375,
        }
    },
    'qwen3-235b-a22b': {
        'bcb_score': {
            'func_score': 48.86,
            'if_score': {
                '1_inst': 83.95,
                '3_inst': 80.38,
                '5_inst': 78.63,
            }
        },
        'lcb_score': {
            'func_score': None,
            'if_score': {
                '1_inst': None,
                '3_inst': None,
                '5_inst': None,
            }
        },
        'arena_rating': {
            'default': 1423,
            'remove_style_control': 1392,
        }
    },
    'qwen3-32b': {
        'bcb_score': {
            'func_score': 47.63,
            'if_score': {
                '1_inst': 76.75,
                '3_inst': 71.81,
                '5_inst': 71.35,
            }
        },
        'lcb_score': {
            'func_score': None,
            'if_score': {
                '1_inst': None,
                '3_inst': None,
                '5_inst': None,
            }
        },
        'arena_rating': {
            'default': 1407,
            'remove_style_control': 1375,
        }
    },
    'qwen3-30b-a3b': {
        'bcb_score': {
            'func_score': 46.40,
            'if_score': {
                '1_inst': 73.42,
                '3_inst': 70.79,
                '5_inst': 69.81,
            }
        },
        'lcb_score': {
            'func_score': 72.42,
            'if_score': {
                '1_inst': 67.77,
                '3_inst': 63.76,
                '5_inst': 63.00,
            }
        },
        'arena_rating': {
            'default': 1378,
            'remove_style_control': 1346,
        }
    },
    'qwen2.5-72b-instruct': {
        'bcb_score': {
            'func_score': 44.39,
            'if_score': {
                '1_inst': 73.68,
                '3_inst': 71.78,
                '5_inst': 70.33,
            }
        },
        'lcb_score': {
            'func_score': 39.05,
            'if_score': {
                '1_inst': 64.83,
                '3_inst': 65.97,
                '5_inst': 66.14,
            }
        },
        'arena_rating': {
            'default': 1346,
            'remove_style_control': 1298,
        }
    },
    'qwen2.5-coder-32b-instruct': {
        'bcb_score': {
            'func_score': 49.39,
            'if_score': {
                '1_inst': 71.40,
                '3_inst': 67.57,
                '5_inst': 65.77,
            }
        },
        'lcb_score': {
            'func_score': None,
            'if_score': {
                '1_inst': None,
                '3_inst': None,
                '5_inst': None,
            }
        },
        'arena_rating': {
            'default': 1325,
            'remove_style_control': 1274,
        }
    },
    'gemma-3-27b-it': {
        'bcb_score': {
            'func_score': 45.53,
            'if_score': {
                '1_inst': 68.42,
                '3_inst': 65.50,
                '5_inst': 65.02,
            }
        },
        'lcb_score': {
            'func_score': 35.92,
            'if_score': {
                '1_inst': 61.99,
                '3_inst': 62.53,
                '5_inst': 63.56,
            }
        },
        'arena_rating': {
            'default': 1370,
            'remove_style_control': 1348,
        }
    },
    'gemma-3-12b-it': {
        'bcb_score': {
            'func_score': 45.53,
            'if_score': {
                '1_inst': 65.96,
                '3_inst': 65.44,
                '5_inst': 65.00,
            }
        },
        'lcb_score': {
            'func_score': 29.29,
            'if_score': {
                '1_inst': 61.33,
                '3_inst': 62.46,
                '5_inst': 62.29,
            }
        },
        'arena_rating': {
            'default': 1332,
            'remove_style_control': 1309,
        }
    },
    'mistral-medium-2505': {
        'bcb_score': {
            'func_score': 45.44,
            'if_score': {
                '1_inst': 73.60,
                '3_inst': 71.02,
                '5_inst': 70.54,
            }
        },
        'lcb_score': {
            'func_score': 40.66,
            'if_score': {
                '1_inst': 62.37,
                '3_inst': 61.90,
                '5_inst': 62.45,
            }
        },
        'arena_rating': {
            'default': 1421,
            'remove_style_control': 1386,
        }
    },
    'minimax-m1': {
        'bcb_score': {
            'func_score': 48.68,
            'if_score': {
                '1_inst': 74.12,
                '3_inst': 71.70,
                '5_inst': 70.89,
            }
        },
        'lcb_score': {
            'func_score': None,
            'if_score': {
                '1_inst': None,
                '3_inst': None,
                '5_inst': None,
            }
        },
        'arena_rating': {
            'default': 1409,
            'remove_style_control': 1368,
        }
    },
    'kimi-k2-0711-preview': {
        'bcb_score': {
            'func_score': 47.19,
            'if_score': {
                '1_inst': 85.00,
                '3_inst': 81.46,
                '5_inst': 79.14,
            }
        },
        'lcb_score': {
            'func_score': 63.58,
            'if_score': {
                '1_inst': 62.75,
                '3_inst': 64.80,
                '5_inst': 64.76,
            }
        },
        'arena_rating': {
            'default': 1454,
            'remove_style_control': 1391,
        }
    },
}


def extract_data(scores_dict, benchmark='bigcodebench', if_instructions='1_inst', arena_type='default'):
    """
    Extract data arrays from scores dictionary

    Args:
        scores_dict: Dictionary containing model scores
        benchmark: 'bigcodebench' or 'livecodebench'
        if_instructions: '1_inst', '3_inst', or '5_inst'
        arena_type: 'default' or 'remove_style_control'
    """
    all_models = list(scores_dict.keys())

    # Map benchmark names to score keys
    benchmark_key = 'bcb_score' if benchmark == 'bigcodebench' else 'lcb_score'

    # Extract scores
    valid_models = []
    func_scores = []
    if_scores = []
    arena_scores = []

    for model in all_models:
        model_data = scores_dict[model]

        # Get functional score
        func_score = model_data[benchmark_key]['func_score']

        # Get instruction following score
        if_score = model_data[benchmark_key]['if_score'][if_instructions]

        # If either score is None, skip this model for the current analysis
        if func_score is None or if_score is None:
            continue

        # Get arena score
        arena_score = model_data['arena_rating'][arena_type]

        # Add model and its scores to the lists
        valid_models.append(model)
        func_scores.append(func_score)
        if_scores.append(if_score)
        arena_scores.append(arena_score)

    return (valid_models,
            np.array(func_scores),
            np.array(if_scores),
            np.array(arena_scores))


def calculate_correlations(x, y, include_kendall=True):
    """Calculate Pearson, Spearman, and (optionally) Kendall correlations using scipy.stats"""
    # Pearson correlation using scipy
    pearson_corr, _ = pearsonr(x, y)

    # Spearman correlation using scipy
    spearman_corr, _ = spearmanr(x, y)

    if not include_kendall:
        return pearson_corr, spearman_corr

    # Kendall tau correlation using scipy
    kendall_corr, _ = kendalltau(x, y)

    return pearson_corr, spearman_corr, kendall_corr


def create_visualizations(results, benchmark='bigcodebench'):
    """Create three elegant line plots with gradient backgrounds (PNG format).

    Reproduces the behavior of the original ``calculate_correlations.py``:
    individual per-correlation PNGs plus a 3-panel combined PNG, written to
    ``figures/{benchmark}/``.
    """
    # Create figures directory based on benchmark
    figures_dir = f'figures/{benchmark}'
    os.makedirs(figures_dir, exist_ok=True)

    # Extract data for plotting
    ratios = np.array([result['ratio'] for result in results])
    # Use instruction following ratio (1 - functional ratio)
    if_ratios = 1 - ratios
    correlation_types = ['pearson', 'spearman', 'kendall']

    # Color schemes for each correlation type
    color_schemes = {
        'pearson': 'viridis',
        'spearman': 'viridis',
        'kendall': 'viridis'
    }

    # Create individual elegant line plots
    for corr_type in correlation_types:
        correlations = np.array([result['arena_correlations'][corr_type] for result in results])

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Create gradient background
        cmap = plt.get_cmap(color_schemes[corr_type])

        # Create smooth interpolation for better gradient effect
        if_smooth = np.linspace(0, 1, 300)
        corr_smooth = np.interp(if_smooth, if_ratios, correlations)

        # Create gradient background using imshow
        gradient = np.linspace(correlations.min(), correlations.max(), 256).reshape(256, 1)
        gradient = np.repeat(gradient, 100, axis=1)

        extent = [0, 1, correlations.min(), correlations.max()]
        ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=0.3, extent=extent, origin='lower')

        # Plot the main line with enhanced styling
        ax.plot(if_ratios, correlations, color='black', linewidth=3, alpha=1.0)
        ax.plot(if_ratios, correlations, color='white', linewidth=1, alpha=0.9)

        # Add scatter points for each data point
        scatter_colors = cmap(plt.Normalize(correlations.min(), correlations.max())(correlations))
        ax.scatter(if_ratios, correlations, c=scatter_colors, s=80, alpha=0.8,
                  edgecolors='white', linewidth=2, zorder=7)

        # Find and highlight optimal point
        max_corr_idx = np.argmax(correlations)
        optimal_if_ratio = if_ratios[max_corr_idx]
        optimal_corr = correlations[max_corr_idx]
        optimal_functional_ratio = ratios[max_corr_idx]

        # Highlight optimal point with special marker
        ax.scatter([optimal_if_ratio], [optimal_corr], s=400, c='gold', marker='*',
                  edgecolors='darkred', linewidth=3, zorder=10, label=f'Optimal: {optimal_corr:.4f}')

        # Add elegant annotation for optimal point
        annotation_text = f'Peak Performance\n{optimal_corr:.4f}\nIF: {optimal_if_ratio:.1f} | Functional: {optimal_functional_ratio:.1f}'
        ax.annotate(annotation_text,
                   xy=(optimal_if_ratio, optimal_corr),
                   xytext=(optimal_if_ratio + 0.15, optimal_corr - 0.01),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                   ha='center', va='top')

        # Customize axes and labels
        ax.set_xlabel('Instruction Following Ratio', fontsize=16, fontweight='bold', labelpad=15)
        ax.set_ylabel('Correlation Coefficient', fontsize=16, fontweight='bold', labelpad=15)

        # Create title
        # title = f'{corr_type.capitalize()} Correlation Analysis\nCorrelations with Chatbot Arena Rating'
        # ax.set_title(title, fontsize=18, fontweight='bold', pad=25)

        # Set axis limits with some padding
        ax.set_xlim(-0.05, 1.05)
        y_range = correlations.max() - correlations.min()
        ax.set_ylim(correlations.min() - y_range*0.05, correlations.max() + y_range*0.05)

        # Enhance grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)

        # Add subtle corner annotations
        ax.text(0.02, 0.98, 'Pure Functional\nFocus', transform=ax.transAxes, fontsize=11,
               fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.7))

        ax.text(0.98, 0.98, 'Pure Instruction\nFollowing Focus', transform=ax.transAxes, fontsize=11,
               fontweight='bold', va='top', ha='right',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))

        # Enhance tick formatting
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.1)])

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(correlations.min(), correlations.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Correlation Strength', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # Add legend
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

        # Style the plot area
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        plt.tight_layout()

        # Save individual plot
        individual_path = os.path.join(figures_dir, f'{corr_type}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Saved {corr_type} plot: {individual_path}")

    # Create combined plot with all three correlations
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Calculate the maximum y-range (delta y) across all correlation types
    max_y_range = 0
    correlation_data = {}

    for corr_type in correlation_types:
        correlations = np.array([result['arena_correlations'][corr_type] for result in results])
        correlation_data[corr_type] = correlations
        y_range = correlations.max() - correlations.min()
        max_y_range = max(max_y_range, y_range)

    for idx, corr_type in enumerate(correlation_types):
        ax = axes[idx]
        correlations = correlation_data[corr_type]
        # Use viridis colormap for all three plots (spearman's colormap)
        cmap = plt.get_cmap('viridis')

        # Calculate unified y-range for this plot
        data_center = (correlations.min() + correlations.max()) / 2
        unified_y_min = data_center - max_y_range / 2 - max_y_range * 0.05
        unified_y_max = data_center + max_y_range / 2 + max_y_range * 0.05

        # Gradient background - extend to cover full y-axis range
        gradient = np.linspace(unified_y_min, unified_y_max, 256).reshape(256, 1)
        gradient = np.repeat(gradient, 100, axis=1)
        extent = [0, 1, unified_y_min, unified_y_max]
        ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=0.3, extent=extent, origin='lower')

        # Main line - use black color as suggested
        ax.plot(if_ratios, correlations, color='black', linewidth=3, alpha=1.0)
        ax.plot(if_ratios, correlations, color='white', linewidth=1, alpha=0.9)

        # Scatter points
        scatter_colors = cmap(plt.Normalize(correlations.min(), correlations.max())(correlations))
        ax.scatter(if_ratios, correlations, c=scatter_colors, s=60, alpha=0.8,
                  edgecolors='white', linewidth=1.5, zorder=7)

        # Optimal point
        max_corr_idx = np.argmax(correlations)
        optimal_if_ratio = if_ratios[max_corr_idx]
        optimal_corr = correlations[max_corr_idx]
        optimal_functional_ratio = ratios[max_corr_idx]

        ax.scatter([optimal_if_ratio], [optimal_corr], s=200, c='gold', marker='*',
                  edgecolors='darkred', linewidth=2, zorder=10)

        # Labels and formatting
        ax.set_xlabel('Instruction Following Ratio', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_title(f'{corr_type.capitalize()}\nOptimal: {optimal_if_ratio:.1f} × IF + {optimal_functional_ratio:.1f} × Func',
                    fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlim(-0.02, 1.02)

        # Use same y-range (delta y) for all plots, centered around each plot's data
        ax.set_ylim(unified_y_min, unified_y_max)

    # Create overall title
    # plt.suptitle('Correlations with Chatbot Arena Rating', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    combined_path = os.path.join(figures_dir, 'overall_correlations.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved combined plot: {combined_path}")


def create_pdf_visualization(results, benchmark='bigcodebench', if_instructions='1_inst', arena_type='default'):
    """
    Creates a side-by-side PDF plot in the exact style of the original script's overall plot.
    This version has EXTRA LARGE fonts for maximum readability.

    Reproduces the behavior of the original ``calculate_two_correlations.py``:
    a single 2-panel (Pearson + Spearman) large-font PDF, written to
    ``figures_pdf/{benchmark}/``.
    """
    figures_dir = f'figures_pdf/{benchmark}'
    os.makedirs(figures_dir, exist_ok=True)

    ratios = np.array([result['ratio'] for result in results])
    if_ratios = 1 - ratios
    correlation_types = ['pearson', 'spearman']

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    correlation_data = {corr_type: np.array([r['arena_correlations'][corr_type] for r in results]) for corr_type in correlation_types}

    max_y_range = 0
    for corr_type in correlation_types:
        correlations = correlation_data[corr_type]
        y_range = correlations.max() - correlations.min()
        max_y_range = max(max_y_range, y_range)

    for idx, corr_type in enumerate(correlation_types):
        ax = axes[idx]
        correlations = correlation_data[corr_type]
        cmap = plt.get_cmap('viridis')

        # Calculate centered y-axis for this specific plot
        data_center = (correlations.min() + correlations.max()) / 2
        unified_y_min = data_center - max_y_range / 2 - max_y_range * 0.05
        unified_y_max = data_center + max_y_range / 2 + max_y_range * 0.05

        # Gradient Background
        gradient = np.linspace(unified_y_min, unified_y_max, 256).reshape(-1, 1)
        extent = [0, 1, unified_y_min, unified_y_max]
        ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=0.3, extent=extent, origin='lower')

        # Styled lines (black with white outline)
        ax.plot(if_ratios, correlations, color='black', linewidth=4, alpha=1.0)
        ax.plot(if_ratios, correlations, color='white', linewidth=1.5, alpha=0.9)

        # Scatter points (enlarged)
        scatter_colors = cmap(plt.Normalize(correlations.min(), correlations.max())(correlations))
        ax.scatter(if_ratios, correlations, c=scatter_colors, s=200, alpha=0.9,
                   edgecolors='white', linewidth=2, zorder=7)

        # Optimal Point Star (enlarged) and Title Text
        max_corr_idx = np.argmax(correlations)
        optimal_if_ratio = if_ratios[max_corr_idx]
        optimal_corr = correlations[max_corr_idx]
        optimal_functional_ratio = ratios[max_corr_idx]

        ax.scatter([optimal_if_ratio], [optimal_corr], s=800, c='gold', marker='*',
                   edgecolors='darkred', linewidth=2.5, zorder=10)

        # --- FONT SIZES INCREASED AGAIN ---
        title_text = f'{corr_type.capitalize()}\nOptimal: {optimal_if_ratio:.1f} × IF + {optimal_functional_ratio:.1f} × Func'
        ax.set_title(title_text, fontsize=36, fontweight='bold', pad=25)

        ax.set_xlabel('Instruction Following Ratio', fontsize=36, fontweight='bold', labelpad=20)
        if idx == 0:
            ax.set_ylabel('Correlation Coefficient', fontsize=36, fontweight='bold', labelpad=20)

        # Tick number font size slightly increased for balance
        ax.tick_params(axis='both', which='major', labelsize=16)

        # General plot settings
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(unified_y_min, unified_y_max)

        # Full box frame
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout()

    # Save as PDF
    pdf_path = os.path.join(figures_dir, f'{if_instructions}_{arena_type}_correlations.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✅ Saved correlation PDF: {pdf_path}")


def benchmark_name(benchmark_key):
    """Human-readable benchmark name."""
    return 'BigCodeBench' if benchmark_key == 'bigcodebench' else 'LiveCodeBench'


def analyze_correlations(scores_dict, benchmark='bigcodebench', if_instructions='1_inst',
                        arena_type='default', ratio_step=0.1, visualization=True,
                        fmt='png'):
    """
    Main analysis function with configurable parameters

    Args:
        scores_dict: Dictionary containing model scores
        benchmark: 'bigcodebench' or 'livecodebench'
        if_instructions: '1_inst', '3_inst', or '5_inst'
        arena_type: 'default' or 'remove_style_control'
        ratio_step: Step size for ratio testing
        visualization: Whether to create visualizations
        fmt: 'png' (Pearson+Spearman+Kendall, 3-panel PNG) or
             'pdf' (Pearson+Spearman, 2-panel large-font PDF)
    """
    include_kendall = (fmt == 'png')

    # Extract data
    models, func_scores, if_scores, arena_scores = extract_data(
        scores_dict, benchmark, if_instructions, arena_type
    )

    name = benchmark_name(benchmark)

    if include_kendall:
        print("=== AI MODEL CORRELATION ANALYSIS ===")
    else:
        print("=== AI MODEL CORRELATION ANALYSIS (Pearson & Spearman) ===")
    print(f"Benchmark: {name}")
    print(f"Instructions: {if_instructions}")
    print(f"Arena type: {arena_type}")
    if include_kendall:
        print(f"Valid Models for Analysis: {len(models)}")
        print(f"Ratio step: {ratio_step}")
        print("Weighting formula: weighted_score = func_weight × func_score + if_weight × if_score\n")
    else:
        print(f"Valid Models: {len(models)}")

    # Define weight ratios to test
    ratios = np.arange(0.0, 1.0 + ratio_step, ratio_step)

    # Store results
    results = []

    # Calculate correlations for each ratio
    for ratio in ratios:
        # Calculate weighted scores: ratio * func_score + (1 - ratio) * if_score
        weighted_scores = ratio * func_scores + (1 - ratio) * if_scores

        # Calculate correlations with arena scores
        if include_kendall:
            arena_pearson, arena_spearman, arena_kendall = calculate_correlations(
                weighted_scores, arena_scores, include_kendall=True)
            arena_correlations = {
                'pearson': arena_pearson,
                'spearman': arena_spearman,
                'kendall': arena_kendall
            }
        else:
            arena_pearson, arena_spearman = calculate_correlations(
                weighted_scores, arena_scores, include_kendall=False)
            arena_correlations = {
                'pearson': arena_pearson,
                'spearman': arena_spearman
            }

        # Store results
        results.append({
            'ratio': ratio,
            'func_weight': ratio,
            'if_weight': 1 - ratio,
            'arena_correlations': arena_correlations
        })

    # Display results using PrettyTable
    table = PrettyTable()
    table.float_format = ".4"
    if include_kendall:
        table.field_names = ["Func Weight", "IF Weight", "Pearson", "Spearman", "Kendall"]
        for result in results:
            table.add_row([
                f"{result['func_weight']:.1f}",
                f"{result['if_weight']:.1f}",
                f"{result['arena_correlations']['pearson']:.4f}",
                f"{result['arena_correlations']['spearman']:.4f}",
                f"{result['arena_correlations']['kendall']:.4f}"
            ])
        print(f"\n{name} ({if_instructions}) vs Arena ({arena_type}) Correlations:")
    else:
        table.field_names = ["Func Weight", "IF Weight", "Pearson", "Spearman"]
        for result in results:
            table.add_row([
                f"{result['func_weight']:.1f}",
                f"{result['if_weight']:.1f}",
                f"{result['arena_correlations']['pearson']:.4f}",
                f"{result['arena_correlations']['spearman']:.4f}"
            ])
        print(f"\nCorrelations:")
    print(table)

    # Create visualizations if requested
    if visualization:
        if fmt == 'png':
            print(f"\nCreating visualizations...")
            create_visualizations(results, benchmark)
            print(f"Visualization completed.")
        else:
            print("\nCreating PDF visualization...")
            create_pdf_visualization(results, benchmark, if_instructions, arena_type)

    return results


def find_optimal_correlations(results, fmt='png'):
    """Find and display optimal correlations and their corresponding ratios"""
    # Find maximum correlations for each type
    if fmt == 'png':
        corr_types = ['pearson', 'spearman', 'kendall']
    else:
        corr_types = ['pearson', 'spearman']

    max_correlations = {corr_type: {'value': -1, 'ratio': 0} for corr_type in corr_types}

    for result in results:
        for corr_type in corr_types:
            corr_val = result['arena_correlations'][corr_type]

            if corr_val > max_correlations[corr_type]['value']:
                max_correlations[corr_type] = {'value': corr_val, 'ratio': result['ratio']}

    print("\n=== OPTIMAL CORRELATIONS ===")

    for corr_type, data in max_correlations.items():
        func_ratio = data['ratio']
        if_ratio = 1 - data['ratio']
        print(f"{corr_type.capitalize():8}: {data['value']:.4f} at {if_ratio:.1f} × IF + {func_ratio:.1f} × Func")


def save_results_to_csv(results, benchmark='bigcodebench', if_instructions='1_inst',
                        arena_type='default', fmt='png'):
    """Save results to CSV file for further analysis.

    The ``png`` format includes the ``kendall`` column; the ``pdf`` format does not.
    """
    # Create directory if it doesn't exist
    results_dir = f'correlation_results/{benchmark}'
    os.makedirs(results_dir, exist_ok=True)

    filename = f'{results_dir}/{if_instructions}_{arena_type}_correlations.csv'

    data = []
    for result in results:
        row = {
            'func_weight': result['func_weight'],
            'if_weight': result['if_weight'],
            'pearson': result['arena_correlations']['pearson'],
            'spearman': result['arena_correlations']['spearman'],
        }
        if fmt == 'png':
            row['kendall'] = result['arena_correlations']['kendall']
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to {filename}")


def main(benchmark='bigcodebench', if_instructions='1_inst', arena_type='default',
         ratio_step=0.1, visualization=True, save_csv=True, fmt='png'):
    """
    Main execution function with configurable parameters

    Args:
        benchmark: 'bigcodebench' or 'livecodebench'
        if_instructions: '1_inst', '3_inst', or '5_inst'
        arena_type: 'default' or 'remove_style_control'
        ratio_step: Step size for ratio testing
        visualization: Whether to create visualizations
        save_csv: Whether to save results to CSV
        fmt: 'png' (3-panel PNG incl. Kendall) or 'pdf' (2-panel large-font PDF)
    """
    # Validate parameters
    valid_benchmarks = ['bigcodebench', 'livecodebench']
    valid_instructions = ['1_inst', '3_inst', '5_inst']
    valid_arena_types = ['default', 'remove_style_control']
    valid_formats = ['png', 'pdf']

    if fmt not in valid_formats:
        print(f"❌ Invalid format: {fmt}. Must be one of {valid_formats}")
        return

    if benchmark not in valid_benchmarks:
        print(f"❌ Invalid benchmark: {benchmark}. Must be one of {valid_benchmarks}")
        return

    if if_instructions not in valid_instructions:
        print(f"❌ Invalid if_instructions: {if_instructions}. Must be one of {valid_instructions}")
        return

    if arena_type not in valid_arena_types:
        print(f"❌ Invalid arena_type: {arena_type}. Must be one of {valid_arena_types}")
        return

    if not (0 < ratio_step <= 1):
        print(f"❌ Invalid ratio_step: {ratio_step}. Must be between 0 and 1")
        return

    # Run the correlation analysis
    results = analyze_correlations(scores, benchmark=benchmark, if_instructions=if_instructions,
                                 arena_type=arena_type, ratio_step=ratio_step,
                                 visualization=visualization, fmt=fmt)

    # Find and display optimal correlations
    find_optimal_correlations(results, fmt=fmt)

    # Save results to CSV
    if save_csv:
        save_results_to_csv(results, benchmark, if_instructions, arena_type, fmt=fmt)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='AI Model Scores Correlation Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--format',
        dest='format',
        type=str,
        choices=['png', 'pdf'],
        default='png',
        help="Output format: 'png' reproduces the 3-panel PNG (incl. Kendall) under "
             "figures/{benchmark}/; 'pdf' reproduces the 2-panel large-font PDF under "
             "figures_pdf/{benchmark}/ (Pearson & Spearman only)."
    )

    parser.add_argument(
        '--benchmark',
        type=str,
        choices=['bigcodebench', 'livecodebench'],
        default='bigcodebench',
        help='Benchmark to use for analysis'
    )

    parser.add_argument(
        '--if_instructions',
        type=str,
        choices=['1_inst', '3_inst', '5_inst'],
        default='1_inst',
        help='Number of instructions for instruction following evaluation'
    )

    parser.add_argument(
        '--arena_type',
        type=str,
        choices=['default', 'remove_style_control'],
        default='default',
        help='Arena rating type to use'
    )

    parser.add_argument(
        '--ratio_step',
        type=float,
        default=0.1,
        help='Step size for ratio testing (between 0 and 1)'
    )

    parser.add_argument(
        '--visualization',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Create visualizations (use --visualization / --no-visualization)'
    )

    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='Save results to CSV file'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run analysis with command line arguments
    main(
        benchmark=args.benchmark,
        if_instructions=args.if_instructions,
        arena_type=args.arena_type,
        ratio_step=args.ratio_step,
        visualization=args.visualization,
        save_csv=args.save_csv,
        fmt=args.format,
    )
