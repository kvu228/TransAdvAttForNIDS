"""
Tiện ích lõi cho TransAdvAttForNIDS (Việt).

Chứa: ``CustomDataset`` đọc CSV flow + chuẩn hóa min-max; ``load_net`` / ``init_net`` nạp
surrogate (60 fea) và target (66/78 fea); ``normalize_df``; ``rectify_adv_flows`` đồng bộ
đặc trưng phụ thuộc sau khi sửa Level-1; ``STORAGE_DIR`` (cần trùng môi trường khi chạy).

Utility functions and classes for TransAdvAttForNIDS project.

This module provides:
- Dataset loading and preprocessing utilities
- Model initialization and loading functions
- Feature normalization functions
- Flow rectification functions for adversarial attack traffic
"""

from torch.utils.data import Dataset
import pandas as pd
import torch
from utils.surrogate_models import mlp_s, cnn_s, ResCNN_s, lstm_s, SelfAttention_s
from utils.target_models import mlp_t, cnn_t, ResCNN_t, lstm_t, SelfAttention_t
from utils.target_models_with_78_fea import mlp_t_78, cnn_t_78, ResCNN_t_78, lstm_t_78, SelfAttention_t_78
from utils.surrogate_model_with_var_input_fea import mlp_s_varfea

# Global storage directory path for datasets and models
# Must be configured before running any scripts
STORAGE_DIR = '/mmlab_students/storageStudents/nguyenvd/nids/TransAdvAttForNIDS/storage'

def normalize_df(df, df_minmax):
    """
    Normalize a DataFrame using min-max normalization.
    
    Normalizes all values in the DataFrame to the range [0, 1] using the formula:
    normalized = (value - min) / (max - min)
    
    Args:
        df (pd.DataFrame): DataFrame to normalize. Each column should correspond to a feature.
        df_minmax (pd.DataFrame): DataFrame containing min and max values for normalization.
            - Row 0: Minimum values for each feature
            - Row 1: Maximum values for each feature
            Must have the same columns as df.
    
    Returns:
        pd.DataFrame: Normalized DataFrame with values in range [0, 1].
            NaN values are filled with 0.
    
    Example:
        >>> df_minmax = pd.DataFrame([[0, 10], [100, 200]], columns=['feat1', 'feat2'])
        >>> df = pd.DataFrame([[50, 150], [25, 175]], columns=['feat1', 'feat2'])
        >>> normalize_df(df, df_minmax)
           feat1  feat2
        0    0.5    0.7
        1   0.25   0.875
    """
    return ((df - df_minmax.loc[0]) / (df_minmax.loc[1] - df_minmax.loc[0])).fillna(0)

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading network flow data.
    
    This dataset loads CSV files containing network flow features, normalizes them,
    and converts them to PyTorch tensors. Labels are converted from string format
    ('Benign' or attack type) to binary format (0 for Benign, 1 for Attack).
    
    Attributes:
        data (torch.Tensor): Normalized feature data as PyTorch tensor.
        label (list): List of integer labels (0 for Benign, 1 for Attack).
    
    Example:
        >>> dataset = CustomDataset(
        ...     fp_data='path/to/data.csv',
        ...     fp_minmax='path/to/minmax.csv',
        ...     fp_fea='path/to/features.csv'
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=128)
        >>> for data, labels in dataloader:
        ...     # Training code
    """
    
    def __init__(self, fp_data, fp_minmax, fp_fea):
        """
        Initialize the CustomDataset.
        
        Args:
            fp_data (str): Path to CSV file containing network flow data.
                Must contain a 'Label' column with 'Benign' or attack type names.
            fp_minmax (str): Path to CSV file containing min-max values for normalization.
                Format: 2 rows, row 0 = min values, row 1 = max values.
            fp_fea (str): Path to CSV file containing list of features to use.
                The CSV should have a header row with feature names as columns.
        """
        self.read_csv_and_precess_df(fp_data, fp_minmax, fp_fea)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (data, label) where:
                - data (torch.Tensor): Feature tensor for the sample
                - label (int): Label (0 for Benign, 1 for Attack)
        """
        return self.data[idx], self.label[idx]

    def read_csv_and_precess_df(self, fp_data, fp_minmax, fp_fea):
        """
        Load and preprocess CSV files.
        
        This method:
        1. Loads the data CSV file
        2. Loads min-max values for normalization
        3. Loads the list of features to use
        4. Selects and normalizes the features
        5. Converts to PyTorch tensors
        6. Converts labels from string to integer format
        
        Args:
            fp_data (str): Path to data CSV file.
            fp_minmax (str): Path to min-max CSV file.
            fp_fea (str): Path to features CSV file.
        
        Note:
            This method modifies the instance variables self.data and self.label.
        """
        df_data = pd.read_csv(fp_data, header=0, index_col=False)
        df_minmax = pd.read_csv(fp_minmax, header=0, index_col=False)
        list_col = pd.read_csv(fp_fea, header=0, index_col=False).columns.tolist()
        
        df = df_data[list_col]
        df = normalize_df(df, df_minmax)
        self.data = torch.from_numpy(df.values).float()

        pos = df_data['Label'] == 'Benign'
        df_data.loc[pos, 'label'] = 0
        df_data.loc[~pos, 'label'] = 1
        self.label = df_data['label'].astype(int).values.tolist()

def init_net(model_type, model_name: str):
    """
    Initialize a neural network model without loading weights.
    
    Creates a new model instance based on the model type and architecture name.
    The model is not loaded with any pre-trained weights.
    
    Args:
        model_type (str): Type of model to create.
            - 't': Target NIDS model (66 or 78 features)
            - 's': Surrogate model (60 features)
        model_name (str): Name of the model architecture. Must include the model type suffix.
            For target models: 'mlp_t', 'cnn_t', 'rescnn_t', 'lstm_t', 'Selfattention_t'
            For surrogate models: 'mlp_s', 'cnn_s', 'rescnn_s', 'lstm_s', 'Selfattention_s'
    
    Returns:
        torch.nn.Module: Initialized model (not loaded with weights).
            Returns None if model_type or model_name is not recognized.
    
    Example:
        >>> model = init_net('t', 'mlp_t')
        >>> model = init_net('s', 'cnn_s')
    """
    net = None
    if model_type == 't':
        if model_name == 'mlp_t':
            net = mlp_t()
        if model_name == 'cnn_t':
            net = cnn_t()
        if model_name == 'rescnn_t':
            net = ResCNN_t()
        if model_name == 'lstm_t':
            net = lstm_t()
        if model_name == 'Selfattention_t':
            net = SelfAttention_t()
    if model_type == 's':
        if model_name == 'mlp_s':
            net = mlp_s()
        if model_name == 'cnn_s':
            net = cnn_s()
        if model_name == 'rescnn_s':
            net = ResCNN_s()
        if model_name == 'lstm_s':
            net = lstm_s()
        if model_name == 'Selfattention_s':
            net = SelfAttention_s()
    return net

def load_net(model_name_or_fea_num, fp_model_or_name: str = None, fp_model_opt: str = None):
    """
    Nạp checkpoint đã huấn luyện.

    Hai cách gọi:
        - ``load_net(model_name, fp_model)``: tên dạng ``ids18_mlp_t`` / ``ton_cnn_s`` (khớp chuỗi trong code).
        - ``load_net(fea_num, model_name, fp_model)``: nếu ``fea_num == 78`` thì dùng biến thể
          target 78 chiều từ ``target_models_with_78_fea``; ngược lại target mặc định 66 fea.

    Surrogate (tên chứa ``_..._s``) luôn 60 fea, không qua tham số ``fea_num``.
    """
    fea_num = None
    if fp_model_opt is not None:
        fea_num = model_name_or_fea_num
        model_name, fp_model = fp_model_or_name, fp_model_opt
    else:
        model_name, fp_model = model_name_or_fea_num, fp_model_or_name
    net = None
    if model_name[-1] == 's':
        if '_mlp_s' in model_name:
            net = mlp_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        if '_cnn_s' in model_name:
            net = cnn_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        if '_rescnn_s' in model_name:
            net = ResCNN_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        if '_lstm_s' in model_name:
            net = lstm_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        if '_Selfattention_s' in model_name:
            net = SelfAttention_s()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        return net
    else:
        use_78 = (fea_num == 78)
        if '_mlp_t' in model_name:
            net = mlp_t_78() if use_78 else mlp_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        if '_cnn_t' in model_name:
            net = cnn_t_78() if use_78 else cnn_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        if '_rescnn_t' in model_name:
            net = ResCNN_t_78() if use_78 else ResCNN_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        if '_lstm_t' in model_name:
            net = lstm_t_78() if use_78 else lstm_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        if '_Selfattention_t' in model_name:
            net = SelfAttention_t_78() if use_78 else SelfAttention_t()
            net.load_state_dict(torch.load(fp_model, weights_only=True, map_location="cpu"))
        return net

def get_var(n, x1, x2, rwa_var, x_mean, delta1, delta2, delta_mean):
    """
    Calculate the updated variance after modifying two values in a dataset.
    
    This function computes the new variance when two values (x1, x2) in a dataset
    are modified by deltas (delta1, delta2). Used for recalculating standard deviation
    of features after adversarial perturbations.
    
    Args:
        n (int or pd.Series): Number of samples in the dataset.
        x1 (pd.Series): Original first values (e.g., 'Fwd Pkt Len Max').
        x2 (pd.Series): Original second values (e.g., 'Fwd Pkt Len Min').
        rwa_var (pd.Series): Original variance values (before modification).
        x_mean (pd.Series): Original mean values.
        delta1 (pd.Series): Change applied to x1 (x1_new - x1_old).
        delta2 (pd.Series): Change applied to x2 (x2_new - x2_old).
        delta_mean (pd.Series): Change in mean value (mean_new - mean_old).
    
    Returns:
        pd.Series: Updated variance values after applying modifications.
    
    Note:
        This is a helper function used internally by rectify_adv_flows() to
        recalculate Level 2 features (standard deviations) after modifying
        Level 1 features (max/min packet lengths and IATs).
    """
    var = (2 * (x1 * delta1 - x_mean * delta1 + x2 * delta2 - x_mean * delta2) + (delta1 - delta_mean) ** 2 + (delta2 - delta_mean) ** 2 + (n - 2) * (delta_mean ** 2) + n * rwa_var) / n
    return var

def rectify_adv_flows(adv_flows: pd.DataFrame, flows: pd.DataFrame, pert: pd.DataFrame):
    """
    Rectify adversarial flows by recalculating dependent features.
    
    After modifying Level 1 features (Fwd Pkt Len Max/Min, Fwd IAT Max/Min),
    this function automatically recalculates all dependent features (Level 2 and Level 3)
    to maintain consistency in the network flow statistics.
    
    Feature Hierarchy:
        - Level 1 (Directly modified): Fwd Pkt Len Max/Min, Fwd IAT Max/Min
        - Level 2 (Computed from Level 1): Flow Duration, TotLen Fwd Pkts, 
          Fwd Pkt Len Mean/Std, Fwd IAT Mean/Std, Pkt Len Mean/Std/Var, etc.
        - Level 3 (Computed from Level 2): Flow Byts/s, Flow Pkts/s, etc.
    
    Args:
        adv_flows (pd.DataFrame): Adversarial flows DataFrame (modified in-place).
            Contains flows with modified Level 1 features.
        flows (pd.DataFrame): Original flows DataFrame (not modified).
            Used as reference for calculating differences.
        pert (pd.DataFrame): Perturbation DataFrame containing the changes applied.
            Should have the same columns as adv_flows.
    
    Returns:
        None: The function modifies adv_flows in-place.
    
    Note:
        - Clips values to ensure physical constraints:
          * Fwd Pkt Len Max <= 1460 (MTU)
          * Fwd Pkt Len Min >= 0
          * Fwd IAT Min >= 1
        - Calculates Level 2 features based on differences from original flows
        - Calculates Level 3 features from Level 2 features
        - This function is called automatically by attack methods (MIFGSM, SIM, etc.)
          after applying perturbations
    
    Example:
        >>> # After generating adversarial examples
        >>> adv_flows = flows.copy()
        >>> adv_flows['Fwd Pkt Len Max'] += perturbation
        >>> rectify_adv_flows(adv_flows, flows, pert)
        >>> # adv_flows now has all dependent features recalculated
    """
    # first level

    # adv_flows.loc[adv_flows['Fwd Pkt Len Max'] > 1460, ['Fwd Pkt Len Max']] = adv_flows['Fwd Pkt Len Max'] - pert['Fwd Pkt Len Max']
    adv_flows['Fwd Pkt Len Max'] = adv_flows['Fwd Pkt Len Max'].clip(upper=1460)

    adv_flows['Fwd Pkt Len Min'] = adv_flows['Fwd Pkt Len Min'].clip(lower=0)
    adv_flows['Fwd IAT Min'] = adv_flows['Fwd IAT Min'].clip(lower=1)

    delta1 = adv_flows['Fwd Pkt Len Max'] - flows['Fwd Pkt Len Max']
    delta2 = adv_flows['Fwd Pkt Len Min'] - flows['Fwd Pkt Len Min']
    delta3 = adv_flows['Fwd IAT Max'] - flows['Fwd IAT Max']
    delta4 = adv_flows['Fwd IAT Min'] - flows['Fwd IAT Min']

    # second level
    adv_flows['Flow Duration'] = flows['Flow Duration'] + (delta3 + delta4)
    adv_flows['TotLen Fwd Pkts'] = flows['TotLen Fwd Pkts'] + (delta1 + delta2)
    adv_flows['Fwd Pkt Len Mean'] = flows['Fwd Pkt Len Mean'] + ((delta1 + delta2) / adv_flows['Tot Fwd Pkts'])
    adv_flows['Fwd Pkt Len Std'] = get_var(flows['Tot Fwd Pkts'], flows['Fwd Pkt Len Max'], flows['Fwd Pkt Len Min'], flows['Fwd Pkt Len Std']**2, flows['Fwd Pkt Len Mean'], delta1, delta2, adv_flows['Fwd Pkt Len Mean']-flows['Fwd Pkt Len Mean']) ** (1/2)

    adv_flows['Fwd IAT Tot'] = flows['Fwd IAT Tot'] + (delta3 + delta4)
    adv_flows['Fwd IAT Mean'] = flows['Fwd IAT Mean'] + ((delta3 + delta4) / (flows['Tot Fwd Pkts'] - 1))
    adv_flows['Fwd IAT Std'] = get_var(flows['Tot Fwd Pkts'] - 1, flows['Fwd IAT Max'], flows['Fwd IAT Min'], flows['Fwd IAT Std']**2, flows['Fwd IAT Mean'], delta3, delta4, adv_flows['Fwd IAT Mean']-flows['Fwd IAT Mean']) ** (1/2)

    adv_flows['Pkt Len Mean'] = flows['Pkt Len Mean'] + ((delta1 + delta2) / (flows['Tot Fwd Pkts'] + flows['Tot Bwd Pkts']))
    adv_flows['Pkt Len Std'] = get_var(flows['Tot Fwd Pkts'] + flows['Tot Bwd Pkts'], flows['Fwd Pkt Len Max'], flows['Fwd Pkt Len Min'], flows['Pkt Len Var'], flows['Pkt Len Mean'], delta1, delta2, adv_flows['Pkt Len Mean']-flows['Pkt Len Mean']) ** (1/2)
    adv_flows['Pkt Len Var'] = adv_flows['Pkt Len Std'] ** 2
    adv_flows['Pkt Len Min'] = adv_flows[['Fwd Pkt Len Min', 'Bwd Pkt Len Min']].min(axis=1)
    adv_flows['Pkt Len Max'] = adv_flows[['Fwd Pkt Len Max', 'Bwd Pkt Len Max']].max(axis=1)

    adv_flows['Pkt Size Avg'] = flows['Pkt Size Avg'] + ((delta1 + delta2) / (flows['Tot Fwd Pkts'] + flows['Tot Bwd Pkts']))

    # third level
    adv_flows['Flow Byts/s'] = (adv_flows['TotLen Fwd Pkts'] + adv_flows['TotLen Bwd Pkts']) / (adv_flows['Flow Duration']/1000000)
    adv_flows['Flow Pkts/s'] = (adv_flows['Tot Fwd Pkts'] + adv_flows['Tot Bwd Pkts']) / (adv_flows['Flow Duration']/1000000)
    adv_flows['Fwd Pkts/s'] = adv_flows['Tot Fwd Pkts'] / (adv_flows['Flow Duration']/1000000)
    adv_flows['Bwd Pkts/s'] = adv_flows['Tot Bwd Pkts'] / (adv_flows['Flow Duration']/1000000)

    adv_flows['Subflow Fwd Byts'] = adv_flows['TotLen Fwd Pkts']