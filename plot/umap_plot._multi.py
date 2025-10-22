import json
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde, entropy
import os

def load_feature_vectors(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            feature_vectors = json.load(f)
        return np.array(feature_vectors)
    except FileNotFoundError:
        print(f"[WARN] Feature vector file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"[WARN] Could not decode JSON: {file_path}")
        return None
    except Exception as e:
        print(f"[WARN] Error loading feature vectors from {file_path}: {e}")
        return None

def load_labels_from_source(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = []
        for item in data:
            labels.append(0 if item.get("Source") == "human" else 1)
        return np.array(labels)
    except FileNotFoundError:
        print(f"[WARN] Source info file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"[WARN] Could not decode JSON: {file_path}")
        return None
    except Exception as e:
        print(f"[WARN] Error loading labels from {file_path}: {e}")
        return None

def standardize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def perform_umap(data, n_components=2, random_state=42):
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=20,
        min_dist=0.2,
        init='spectral',
        metric='cosine',
        n_jobs=-1
    )
    embedding = reducer.fit_transform(data)
    return embedding

def visualize_subplots(groups, save_path='umap_labeled_subplots.pdf', figsize=(20, 10), dpi=300):
    """
    groups: list of dicts: {'features': np.array, 'labels': np.array, 'title': str}
    """
    n_groups = len(groups)
    if n_groups == 0:
        print("No groups to plot.")
        return

    rows = 2
    cols = 4
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn')
        except Exception:
            plt.style.use('default')
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    # overall figure styling
    fig.patch.set_facecolor('white')
    #fig.suptitle('UMAP of Feature Groups', fontsize=18, weight='bold')
    plt.subplots_adjust(top=0.92, hspace=0.12, wspace=0.12, left=0.05, right=0.98)

    for i in range(rows * cols):
        ax = axes[i]
        # default clean axes
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_facecolor('#fbfbfb')
        # Only set subplot borders to make them more visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#333333')   # Dark gray / near black
            spine.set_linewidth(1.2)        # Slightly thicker for visibility
            spine.set_alpha(0.95)


        ax.set_axisbelow(True)  
        ax.grid(True, which='major', color='k', linewidth=0.5, alpha=0.1, linestyle='--')

        # ax.minorticks_on()
        # ax.grid(True, which='minor', color='k', linestyle=':', linewidth=0.3, alpha=0.25)

        if i >= n_groups:
            ax.set_visible(False)
            continue

        grp = groups[i]
        X = grp['features']
        labels = grp['labels']
        title = grp.get('title', f'Group {i+1}')

        if X is None or labels is None:
            ax.text(0.5, 0.5, f"Missing data\n{title}", ha='center', va='center', fontsize=10, color='gray')
            continue

        if X.shape[0] != labels.shape[0]:
            ax.text(0.5, 0.5, f"Count mismatch\nfeatures:{X.shape[0]} labels:{labels.shape[0]}", ha='center', va='center', fontsize=9, color='gray')
            continue

        # standardize & UMAP
        try:
            X_scaled, _ = standardize_data(X)
            emb = perform_umap(X_scaled, n_components=2)
        except Exception as e:
            ax.text(0.5, 0.5, f"UMAP failed: {e}", ha='center', va='center', fontsize=9, color='red')
            continue

        # color by label
        cmap = {0: 'forestgreen', 1: 'firebrick'}
        colors = [cmap.get(int(l), '#888888') for l in labels]
        # Increase zorder when plotting points to ensure points are above grid
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.9, s=18, linewidths=0, rasterized=True, zorder=4)

        # title and small subtitle with counts
        n_h = int((labels == 0).sum())
        n_g = int((labels == 1).sum())
        ax.set_title(f"{title}", fontsize=10, weight='semibold', pad=6)
        #ax.text(0.98, 0.02, f"H:{n_h}  G:{n_g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray')

        # add legend box in upper-left of each subplot
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label='Human Written Text', markerfacecolor='forestgreen', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='LLM Generated Text', markerfacecolor='firebrick', markersize=6),
        ]
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=8, frameon=True, framealpha=0.9, handletextpad=0.4)

    # final save
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    print(f"Saved multi-panel UMAP to: {save_path}")
    plt.close(fig)


# ...existing code...

def calculate_distribution_distances(dist1_pdf_values, dist2_pdf_values):
    p = np.maximum(dist1_pdf_values, 0)
    q = np.maximum(dist2_pdf_values, 0)
    epsilon = 1e-10
    P_norm = (p + epsilon) / np.sum(p + epsilon)
    Q_norm = (q + epsilon) / np.sum(q + epsilon)
    kl_pq = entropy(P_norm, Q_norm)
    kl_qp = entropy(Q_norm, P_norm)
    hellinger_dist = np.sqrt(np.sum((np.sqrt(P_norm) - np.sqrt(Q_norm))**2)) / np.sqrt(2)
    return kl_pq, kl_qp, hellinger_dist

if __name__ == "__main__":
    # Configuration for different groups
    groups_config = [
        ('Yelp_result/feature_vectors.json', 'Yelp_result/rewrite_data.json', '(a)Yelp'),
        ('News_result/feature_vectors.json', 'News_result/rewrite_data.json', '(b)News'),
        ('Code_result/feature_vectors.json', 'Code_result/rewrite_data.json', '(c)Code'),
        ('Essay_result/feature_vectors.json', 'Essay_result/rewrite_data.json', '(d)Essay'),
        ('perturbed_result/yelp_feature_vectors.json', 'perturbed_result/yelp_rewrite_data.json', '(e)Yelp(Perturbed)'),
        ('perturbed_result/news_feature_vectors.json', 'perturbed_result/news_rewrite_data.json', '(f)News(Perturbed)'),
        ('perturbed_result/code_feature_vectors.json', 'perturbed_result/code_rewrite_data.json', '(g)Code(Perturbed)'),
        ('perturbed_result/essay_feature_vectors.json', 'perturbed_result/essay_rewrite_data.json', '(h)Essay(Perturbed)')
        # ('Yelp_result/feature_vectors.json', 'Yelp_result/rewrite_data.json', '(a)Yelp'),
        # ('News_result/feature_vectors.json', 'News_result/rewrite_data.json', '(b)News'),
        # ('Code_result/feature_vectors.json', 'Code_result/rewrite_data.json', '(c)Code'),
        # ('Essay_result/feature_vectors.json', 'Essay_result/rewrite_data.json', '(d)Essay'),
        # ('RAID_result/mix1_feature_vectors.json', 'RAID_result/mix1_rewrite_data.json', '(e)RAID Mix1'),
        # ('RAID_result/mix2_feature_vectors.json', 'RAID_result/mix2_rewrite_data.json', '(f)RAID Mix2'),
        # ('RAID_result/att1_feature_vectors.json', 'RAID_result/att1_rewrite_data.json', '(g)RAID Att1'),
        # ('RAID_result/att2_feature_vectors.json', 'RAID_result/att2_rewrite_data.json', '(h)RAID Att2')

    ]

    groups = []
    for (fv_path, src_path, title) in groups_config[:8]:
        fv = load_feature_vectors(fv_path)
        labels = load_labels_from_source(src_path)
        groups.append({'features': fv, 'labels': labels, 'title': title})

    out_pdf = 'umap_labeled_subplots.pdf'
    visualize_subplots(groups, save_path=out_pdf)

    # Calculate distribution distances for each group
    for idx, g in enumerate(groups):
        fv = g['features']
        labels = g['labels']
        if fv is None or labels is None or fv.shape[0] != labels.shape[0]:
            print(f"[INFO] Skipping distance calc for group {idx} ({g['title']}) due to missing/mismatched data.")
            continue
        try:
            scaled, _ = standardize_data(fv)
            emb = perform_umap(scaled, n_components=2)
            human_emb = emb[labels == 0]
            gpt_emb = emb[labels == 1]
            if len(human_emb) < 2 or len(gpt_emb) < 2:
                print(f"[INFO] Not enough points for KDE in group {idx} ({g['title']}).")
                continue
            kde_h = gaussian_kde(human_emb.T)
            kde_g = gaussian_kde(gpt_emb.T)
            # create eval grid
            x_min, y_min = np.min(emb, axis=0) - 0.1
            x_max, y_max = np.max(emb, axis=0) + 0.1
            num_grid = 150
            xg = np.linspace(x_min, x_max, num_grid)
            yg = np.linspace(y_min, y_max, num_grid)
            Xg, Yg = np.meshgrid(xg, yg)
            pts = np.vstack([Xg.ravel(), Yg.ravel()])
            p_vals = kde_h(pts)
            q_vals = kde_g(pts)
            kl_pq, kl_qp, hell = calculate_distribution_distances(p_vals, q_vals)
            print(f"[DIST] Group {idx} ({g['title']}): KL(H||G)={kl_pq:.4f}, KL(G||H)={kl_qp:.4f}, Hellinger={hell:.4f}")
        except Exception as e:
            print(f"[WARN] Distance calc failed for group {idx} ({g['title']}): {e}")