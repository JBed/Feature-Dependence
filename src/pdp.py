from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Constants
DEFAULT_FIGSIZE = (8, 6)
DEFAULT_BINS = 4
DEFAULT_DIGITS = 2

@dataclass
class PDPResult:
    feature: str
    bins: np.ndarray
    mean_values: np.ndarray
    std_values: np.ndarray
    bin_counts: np.ndarray

def pdp(df: pd.DataFrame, 
        features: List[str], 
        yname: str, 
        n: int = DEFAULT_BINS, 
        writefolder: Optional[str] = None, 
        digits: int = DEFAULT_DIGITS, 
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE, 
        showbincount: bool = True, 
        ylim_origin: bool = True,
        even_spaced_ticks: bool = False) -> List[PDPResult]:
    """
    Generate Partial Dependence Plots for given features.

    Args:
        df (pd.DataFrame): Input dataframe
        features (List[str]): List of feature names to plot
        yname (str): Name of the target variable
        n (int): Number of bins for continuous variables
        writefolder (Optional[str]): Folder to save plots, if None plots are displayed
        digits (int): Number of decimal places for bin labels
        figsize (Tuple[int, int]): Figure size for plots
        showbincount (bool): Whether to show bin counts on secondary y-axis
        ylim_origin (bool): Whether y-axis should start from 0
        even_spaced_ticks (bool): Whether to use evenly spaced ticks for continuous variables

    Returns:
        List[PDPResult]: List of PDPResult objects for each feature
    """
    if not isinstance(yname, str):
        raise ValueError('yname must be a string. Unique column.')
    if yname not in df.columns:
        raise ValueError(f'yname column "{yname}" is not in the dataframe.')
    if not isinstance(features, list):
        raise ValueError('features must be a list. If single feature use [feature].')

    results = []

    for feature in features:
        if feature == yname or feature not in df.columns:
            print(f'Skipping feature {feature}')
            continue
        
        df_temp = prepare_data(df, feature, yname)
        bins, bin_counts = create_bins(df_temp, feature, n)
        
        if is_continuous_variable(bins):
            result = process_continuous_feature(df_temp, feature, yname, bins, bin_counts, 
                                                digits, even_spaced_ticks)
        else:
            result = process_categorical_feature(df_temp, feature, yname, bins)
        
        results.append(result)
        
        plot_pdp(result, yname, writefolder, figsize, showbincount, ylim_origin)

    return results


def process_continuous_feature(df: pd.DataFrame, 
                               feature: str, 
                               yname: str, 
                               bins: np.ndarray, 
                               bin_counts: np.ndarray, 
                               digits: int, 
                               even_spaced_ticks: bool) -> PDPResult:
    """
    Process continuous feature and return PDPResult.

    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Name of the feature to process
        yname (str): Name of the target variable
        bins (np.ndarray): Array of bin edges
        bin_counts (np.ndarray): Array of bin counts
        digits (int): Number of decimal places for bin labels
        even_spaced_ticks (bool): Whether to use evenly spaced ticks

    Returns:
        PDPResult: Object containing processed data for plotting
    """
    mean_values = []
    std_values = []
    bin_labels = []
    bin_centers = []

    for i in range(len(bins) - 1):
        mask = (df[feature] >= bins[i]) & (df[feature] < bins[i+1])
        bin_data = df.loc[mask, yname]
        
        if not bin_data.empty and not np.isnan(bin_data.mean()) and not np.isnan(bin_data.std()):
            mean_values.append(bin_data.mean())
            std_values.append(bin_data.std() / np.sqrt(bin_counts[i]))
            
            bin_label = f'[{bins[i]:.{digits}f}-{bins[i+1]:.{digits}f})'
            bin_labels.append(bin_label)
            
            if even_spaced_ticks:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
            else:
                bin_centers.append(i)

    mean_values = np.array(mean_values)
    std_values = np.array(std_values)
    bin_centers = np.array(bin_centers)

    return PDPResult(
        feature=feature,
        bins=np.array(bin_centers),
        mean_values=mean_values,
        std_values=std_values,
        bin_counts=bin_counts[:-1]  # exclude the last bin count as it's not used
    )



def process_categorical_feature(df: pd.DataFrame, 
                                feature: str, 
                                yname: str, 
                                bins: np.ndarray) -> PDPResult:
    """
    Process categorical feature and return PDPResult.

    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Name of the feature to process
        yname (str): Name of the target variable
        bins (np.ndarray): Array of unique categories

    Returns:
        PDPResult: Object containing processed data for plotting
    """
    mean_values = []
    std_values = []
    bin_counts = []

    for category in bins:
        category_data = df[df[feature] == category][yname]
        
        if not category_data.empty:
            mean_values.append(category_data.mean())
            std_values.append(category_data.std() / np.sqrt(len(category_data)))
            bin_counts.append(len(category_data))
        else:
            # Handle empty categories
            mean_values.append(np.nan)
            std_values.append(np.nan)
            bin_counts.append(0)

    # Convert lists to numpy arrays
    mean_values = np.array(mean_values)
    std_values = np.array(std_values)
    bin_counts = np.array(bin_counts)

    # Remove NaN values if any
    valid_indices = ~np.isnan(mean_values)
    bins = bins[valid_indices]
    mean_values = mean_values[valid_indices]
    std_values = std_values[valid_indices]
    bin_counts = bin_counts[valid_indices]

    return PDPResult(
        feature=feature,
        bins=bins,
        mean_values=mean_values,
        std_values=std_values,
        bin_counts=bin_counts
    )

from typing import Optional, Tuple
import os

def plot_pdp(result: PDPResult, 
             yname: str, 
             writefolder: Optional[str], 
             figsize: Tuple[int, int], 
             showbincount: bool, 
             ylim_origin: bool):
    """
    Plot Partial Dependence Plot for a feature.

    Args:
        result (PDPResult): PDPResult object containing data to plot
        yname (str): Name of the target variable
        writefolder (Optional[str]): Folder to save the plot, if None the plot is displayed
        figsize (Tuple[int, int]): Figure size for the plot
        showbincount (bool): Whether to show bin counts on secondary y-axis
        ylim_origin (bool): Whether y-axis should start from 0
    """
    with plt.style.context('seaborn'):  # Use seaborn style
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot mean values with error bars
        ax1.errorbar(result.bins, result.mean_values, yerr=result.std_values, 
                     fmt='o-', capsize=5, label=f'mean {yname}')

        ax1.set_xlabel(result.feature)
        ax1.set_ylabel(f'mean {yname}')

        # Set y-axis limit
        if ylim_origin:
            ax1.set_ylim(bottom=0)
        else:
            y_min = min(0, (result.mean_values - result.std_values).min())
            ax1.set_ylim(bottom=y_min * 1.1)  # 10% below the lowest point

        ax1.set_ylim(top=(result.mean_values + result.std_values).max() * 1.1)  # 10% above the highest point

        # Show bin counts if requested
        if showbincount:
            ax2 = ax1.twinx()
            ax2.plot(result.bins, result.bin_counts, 'r--', label='bin count')
            ax2.set_ylabel('bin count', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(bottom=0, top=max(result.bin_counts) * 1.1)  # 10% above the highest count

        # Set x-axis ticks and labels
        if result.bins.dtype.kind in 'fc':  # float or complex (continuous)
            ax1.set_xticks(result.bins)
            ax1.set_xticklabels([f'{x:.2f}' for x in result.bins], rotation=45, ha='right')
        else:  # categorical
            ax1.set_xticks(range(len(result.bins)))
            ax1.set_xticklabels(result.bins, rotation=45, ha='right')

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if showbincount:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')

        plt.title(f'Partial Dependence Plot: {result.feature} vs {yname}')
        plt.tight_layout()

        # Save or show the plot
        if writefolder:
            filename = f'pdp_{result.feature}_{yname}.png'
            filepath = os.path.join(writefolder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def plot_ice(model, df: pd.DataFrame, feature: str, yname: str, num_lines: int = 50):
    """
    Plot Individual Conditional Expectation (ICE) curves.
    
    Args:
        model: Trained model with a predict method
        df (pd.DataFrame): Input dataframe
        feature (str): Name of the feature to analyze
        yname (str): Name of the target variable
        num_lines (int): Number of ICE lines to plot
    """
    feature_values = np.linspace(df[feature].min(), df[feature].max(), 100)
    
    # Randomly select instances to plot
    instances = df.sample(num_lines)
    
    plt.figure(figsize=(10, 6))
    
    for _, instance in instances.iterrows():
        ice_df = pd.concat([instance] * len(feature_values), axis=1).T
        ice_df[feature] = feature_values
        predictions = model.predict(ice_df)
        plt.plot(feature_values, predictions, alpha=0.1, color='blue')
    
    # Plot the average (PDP)
    pdp = np.mean([model.predict(ice_df) for _, instance in instances.iterrows()], axis=0)
    plt.plot(feature_values, pdp, color='red', linewidth=2, label='PDP')
    
    plt.xlabel(feature)
    plt.ylabel(f'Predicted {yname}')
    plt.title(f'ICE Plot for {feature}')
    plt.legend()
    plt.show()

import numpy as np

def calculate_feature_importance(pdp_result: PDPResult) -> float:
    """
    Calculate feature importance based on the variance of the partial dependence function.
    
    Args:
        pdp_result (PDPResult): PDPResult object containing PDP data
    
    Returns:
        float: Importance score for the feature
    """
    # Calculate the variance of the mean values
    importance = np.var(pdp_result.mean_values)
    
    # Normalize by the range of the target variable
    importance /= (np.max(pdp_result.mean_values) - np.min(pdp_result.mean_values))**2
    
    return importance



def plot_feature_importances(results: List[PDPResult], 
                             figsize: Tuple[int, int] = (10, 6),
                             writefolder: Optional[str] = None):
    """
    Plot feature importances based on PDP results.
    
    Args:
        results (List[PDPResult]): List of PDPResult objects
        figsize (Tuple[int, int]): Figure size for the plot
        writefolder (Optional[str]): Folder to save the plot, if None the plot is displayed
    """
    importances = [result.importance for result in results]
    features = [result.feature for result in results]
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)
    pos = np.arange(len(features))
    
    plt.figure(figsize=figsize)
    plt.barh(pos, np.array(importances)[sorted_idx])
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature Importances based on PDP')
    
    if writefolder:
        filename = 'feature_importances.png'
        filepath = os.path.join(writefolder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
