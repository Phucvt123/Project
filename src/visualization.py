import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numeric(numeric_info,data_clean,simple_kde):
    n_cols = len(numeric_info)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i, (idx, name) in enumerate(numeric_info):
        col_float = data_clean[:, idx].astype(float)
        col_valid = col_float[~np.isnan(col_float)]
        
        ax1 = axes[0, i]
        ax1.boxplot(col_valid, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='red'), whiskerprops=dict(color='black'),
                    capprops=dict(color='black'), flierprops=dict(marker='o', markersize=4))
        ax1.set_title(f'Boxplot: {name}', fontsize=12, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)

        ax2 = axes[1, i]
        ax2.hist(col_valid, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        x_grid, density = simple_kde(col_valid)
        ax2.plot(x_grid, density, 'red', linewidth=2.5, label='KDE')
        
        ax2.set_title(f'Histogram + KDE: {name}', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_categorical(cat_cols,data):
    cols_to_plot = cat_cols 

    n_cols = 2
    n_rows = int(np.ceil(len(cols_to_plot) / n_cols)) 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    axes = axes.flatten()

    for i, (idx, name) in enumerate(cols_to_plot):
        ax = axes[i]
        
        col_data = data[:, idx]
        valid_values = np.array([v if v != '' else 'Missing' for v in col_data])
        
        vals, counts = np.unique(valid_values, return_counts=True)
        
        sorted_indices = np.argsort(-counts)
        vals_sorted = vals[sorted_indices]
        counts_sorted = counts[sorted_indices]
        
        top_n = 10
        if len(vals_sorted) > top_n:
            plot_vals = vals_sorted[:top_n]
            plot_counts = counts_sorted[:top_n]
            title = f'{name} (Top {top_n}/{len(vals_sorted)})'
        else:
            plot_vals = vals_sorted
            plot_counts = counts_sorted
            title = f'Phân bố: {name}'
        
        sns.barplot(x=plot_vals, y=plot_counts, ax=ax)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Số lượng')
        ax.tick_params(axis='x', rotation=45)
            
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points', fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_missing_values(missing_stats,missing_mask,col_names,col_indices_missing):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    names = [x[0] for x in missing_stats]
    pcts = [x[1] for x in missing_stats]

    sorted_indices = np.argsort(pcts)[::-1]
    names_sorted = np.array(names)[sorted_indices]
    pcts_sorted = np.array(pcts)[sorted_indices]

    axes[0].bar(names_sorted, pcts_sorted, color='salmon', edgecolor='black')
    axes[0].set_title('Tỷ lệ thiếu dữ liệu theo cột (%)', fontsize=14)
    axes[0].set_ylabel('Phần trăm (%)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    matrix_missing = missing_mask[:, col_indices_missing].astype(int)
    cols_missing_names = [col_names[i] for i in col_indices_missing]
    axes[1].imshow(matrix_missing, aspect='auto', cmap='binary', interpolation='nearest')
    axes[1].set_title('Bản đồ phân bố dữ liệu thiếu (Vạch đen là bị thiếu)', fontsize=14)
    axes[1].set_xticks(range(len(cols_missing_names)))
    axes[1].set_xticklabels(cols_missing_names, rotation=45, ha='right')
    axes[1].set_xlabel('Các cột bị thiếu dữ liệu')
    axes[1].set_ylabel('Chỉ số dòng (Index)')

    plt.tight_layout()
    plt.show()

def plot_brain_drain(cdi_labels,heatmap_matrix,edu_levels):
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(heatmap_matrix, cmap='YlOrRd')


    for i in range(len(edu_levels)):
        for j in range(len(cdi_labels)):
            text = ax.text(j, i, f"{heatmap_matrix[i, j]:.1f}%",
                        ha="center", va="center", color="black", fontweight='bold')

    ax.set_xticks(np.arange(len(cdi_labels)))
    ax.set_yticks(np.arange(len(edu_levels)))
    ax.set_xticklabels(cdi_labels)
    ax.set_yticklabels(edu_levels)
    ax.set_title("Tỷ lệ nghỉ việc (%) theo Trình độ & Mức độ phát triển TP")
    ax.set_xlabel("City Development Index")
    ax.set_ylabel("Education Level")
    plt.grid(False)
    plt.colorbar(im, label='Churn Rate (%)')
    plt.show()


def plot_startup_vs_bigcorp(labels,results):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    x_pos = np.arange(len(labels))
    plt.plot(x_pos, results['Pvt Ltd'], marker='o', markersize=10, linewidth=3, 
            label='Pvt Ltd (Cty lớn)', color='black', markeredgecolor='white', markeredgewidth=2)


    plt.plot(x_pos, results['Early Stage Startup'], marker='s', markersize=10, linewidth=3, 
            label='Startup (Biến động)', color='red', markeredgecolor='white', markeredgewidth=2)


    for i, val in enumerate(results['Pvt Ltd']):
        plt.annotate(f"{val:.1f}%", 
                    (i, val), 
                    textcoords="offset points", 
                    xytext=(0, -20),
                    ha='center', 
                    color='black', 
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)) 

    for i, val in enumerate(results['Early Stage Startup']):
        plt.annotate(f"{val:.1f}%", 
                    (i, val), 
                    textcoords="offset points", 
                    xytext=(0, 15), 
                    ha='center', 
                    color='red', 
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)) 

    plt.xticks(x_pos, labels, fontsize=12, fontweight='medium')
    plt.yticks(fontsize=11)
    plt.ylabel('Tỷ lệ Nghỉ việc (%)', fontsize=13)
    plt.title('Xu hướng Nghỉ việc theo Thâm niên: Startup vs Big Corp', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(5, 40) 
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)



    plt.tight_layout()
    plt.show()


def plot_training_hours(labels,churn_rates):
    plt.figure(figsize=(10, 6))

    bars = plt.bar(labels, churn_rates, color='blue', edgecolor='black', alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', color='black')

    plt.ylabel('Tỷ lệ Nghỉ việc (%)', fontsize=12)
    plt.xlabel('Số giờ đào tạo', fontsize=12)
    plt.title('Tỷ lệ nghỉ việc theo thời lượng đào tạo', fontsize=14, fontweight='bold')
    plt.ylim(0, 35) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.show()
def plot_confusion_matrix(y_true, y_pred,title):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate([y_true,y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = np.where(classes == t)[0][0]  
        j = np.where(classes == p)[0][0]   
        cm[i, j] += 1
    plt.figure(figsize=(6,5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Ở lại', 'Nghỉ việc'],
                yticklabels=['Ở lại', 'Nghỉ việc'])
    plt.xlabel('Dự báo')
    plt.ylabel('Thực tế')
    plt.title(f'Confusion Matrix for {title}')
    plt.show()