from torch.utils.data import DataLoader
# Use optimized version with caching
from data_provider.data_loader_soh_optimized import Dataset_SOH_Forecasting, Dataset_SOH_to_SOH, Dataset_IC2ML, collate_fn_soh, collate_fn_ic2ml


def data_provider_soh(args, flag, input_mode='current_voltage', label_scaler=None):
    """
    Data provider for SOH trajectory prediction
    
    Args:
        args: Configuration arguments
        flag: 'train', 'val', or 'test'
        input_mode: 'current_voltage' or 'soh_to_soh' or 'capacity_increment'
    """
    
    # Select dataset based on input mode
    if input_mode == 'soh_to_soh':
        Data = Dataset_SOH_to_SOH
    elif input_mode == 'capacity_increment':
        Data = Dataset_IC2ML
    else:
        Data = Dataset_SOH_Forecasting
    
    # Determine if shuffle is needed
    shuffle_flag = (flag == 'train')
    
    # Set batch size
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 32
    
    # Create dataset
    if input_mode == 'soh_to_soh' or input_mode == 'capacity_increment':
        data_set = Data(
            args=args,
            flag=flag,
            max_trajectory_len=args.pred_len if hasattr(args, 'pred_len') else 5000,
            label_scaler=label_scaler
        )
    else:
        data_set = Data(
            args=args,
            flag=flag,
            input_mode=input_mode,
            max_trajectory_len=args.pred_len if hasattr(args, 'pred_len') else 5000,
            label_scaler=label_scaler
        )
    
    # Create dataloader
    if input_mode == 'capacity_increment':
        data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        drop_last=False,
        collate_fn=collate_fn_ic2ml
    )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
            drop_last=False,
            collate_fn=collate_fn_soh
        )
    
    return data_set, data_loader