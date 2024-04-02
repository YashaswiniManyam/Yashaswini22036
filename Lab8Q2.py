import pandas as pd

def equal_width_binning(feature_column, num_bins):
    # Calculate bin width
    min_val = feature_column.min()
    max_val = feature_column.max()
    bin_width = (max_val - min_val) / num_bins
    
    # Perform binning
    bins = [min_val + i * bin_width for i in range(num_bins)]
    labels = [f'Bin_{i}' for i in range(1, num_bins)]  # Adjusted labels
    binned_feature = pd.cut(feature_column, bins=bins, labels=labels, include_lowest=True)
    
    return binned_feature

def equal_frequency_binning(feature_column, num_bins):
    # Calculate bin boundaries
    quantiles = [i / num_bins for i in range(num_bins + 1)]
    bin_boundaries = feature_column.quantile(quantiles)
    
    # Perform binning
    binned_feature = pd.cut(feature_column, bins=bin_boundaries, labels=False, include_lowest=True)
    return binned_feature

def binning(feature_column, num_bins=None, binning_type='equal_width'):
    if binning_type == 'equal_width':
        if num_bins is None:
            num_bins = 10  # Default number of bins
        return equal_width_binning(feature_column, num_bins)
    elif binning_type == 'equal_frequency':
        if num_bins is None:
            num_bins = 10  # Default number of bins
        return equal_frequency_binning(feature_column, num_bins)
    else:
        raise ValueError("Invalid binning type. Choose either 'equal_width' or 'equal_frequency'.")

# Test case
data = pd.read_excel(r"C:\\Users\\YashaswiniManyam\\Machine Leaning\\Lab4\\training_mathbert 1.xlsx")
feature_column = data['embed_0']  # Example feature column

# Perform binning with default parameters (equal width, 10 bins)
binned_feature = binning(feature_column)
print("Binned feature (Equal width, 10 bins):\n", binned_feature)

# Perform binning with specified parameters (equal frequency, 5 bins)
binned_feature = binning(feature_column, num_bins=5, binning_type='equal_frequency')
print("\nBinned feature (Equal frequency, 5 bins):\n", binned_feature)
