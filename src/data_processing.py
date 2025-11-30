import numpy as np

def load_csv(path):
    with open(path,'r') as file:
        col_names = file.readline().strip().split(',')
    data = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    return data,col_names
def write_csv(data,col_names,name_file):
    header_str = ','.join(col_names)
    np.savetxt(name_file,data,delimiter=',',header = header_str,comments='',fmt='%s')
def drop_columns(col_names,data,cols_drop):
    drop_idx = [col_names.index(c) for c in col_names if c in cols_drop]
    keep_idx = [i for i in range(len(col_names)) if i not in drop_idx]

    new_data = data[:,keep_idx]
    new_col_names = [col_names[i] for i in keep_idx]
    return new_data,new_col_names

def fill_missing_numeric(data):
    data_filled = data.astype(float)
    for col_index in range(data.shape[1]):
        col = data_filled[:,col_index]
        median = np.nanmedian(col)
        mask = np.isnan(col)
        col[mask] = median
        data_filled[:,col_index] = col
    return data_filled

def fill_missing_mode(data,col_idx):
    data_filled = data.copy()
    for col_index in col_idx:
        col = data_filled[:,col_index]
        non_empty = col[col != '']
        unique,counts = np.unique(non_empty,return_counts=True)
        mode_val = unique[np.argmax(counts)]
        col[col == ''] = mode_val
        data_filled[:,col_index] = col
    return data_filled
def encode_label(data,col_idx):
    data_encoded = data.copy()
    encoders = {}
    for idx in col_idx:
        col = data_encoded[:, idx].astype(str) 
        uniques, encoded_col = np.unique(col, return_inverse=True)
        data_encoded[:, idx] = encoded_col
        encoders[idx] = {v : i for i,v in enumerate(uniques)}
        
    return data_encoded,encoders

def standard(data,mean=None, std=None):
    data = data.astype(float)
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1.0
    data = (data - mean)/std
    return data,mean,std
def fill_missing_constant(data,col_idx, value = 'Unknown'):
    data_filled = data.copy()
    for idx in col_idx:
        col = data_filled[:,idx]
        mask = col == ''
        col[mask] = value
        data[:,idx] = col
    return data_filled
def encode_columns_with_mapper(data, col_indices, encoders):
    data = np.asarray(data, dtype=object)
    data_encoded = data.copy()
    
    for col_idx in col_indices:
        if col_idx >= data.shape[1]:
            print(f"Warning: Cột index {col_idx} vượt quá số cột dữ liệu!")
            continue
            
        if col_idx not in encoders:
            print(f"Warning: Không tìm thấy mapper cho cột {col_idx} → bỏ qua")
            continue
            
        mapping = encoders[col_idx]                    
    
    
        col = data_encoded[:, col_idx]
        encoded_col = np.vectorize(lambda x: mapping.get(x, -1))(col)
        
        data_encoded[:, col_idx] = encoded_col.astype(np.int32)
    
    return data_encoded