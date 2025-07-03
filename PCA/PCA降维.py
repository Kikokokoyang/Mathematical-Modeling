import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('亚类数据集.xlsx','PCA')
print(df.shape)
print(df.head())
# 选择数据集中的特征列，假设所有列都是特征列，排除了ID和目标变量（如果有的话）
X = df.iloc[:, :-1]  # 特征（提取除最后一列以外的特征）
y = df.iloc[:, -1]   # （不是特征的一部分）

# 数据标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X_scaled.shape)

# 初始化PCA对象，设置主成分数（例如，选择2个主成分）
pca = PCA(n_components=14)
X_pca = pca.fit_transform(X_scaled)

# 转换PCA结果为DataFrame
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14'])
# 打印PCA结果的头部数据
print(X_pca_df.head())

X_pca_df['Sample'] = df.index

# 打印主成分的解释方差比例
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 转换载荷矩阵为DataFrame
components_df = pd.DataFrame(pca.components_.T, index=df.columns[:-1], columns=['PC1', 'PC2', 'PC3',  'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14'])
print(components_df.head())

# --------------------------------绘制载荷矩阵图---------------------
# 将components_df转换为numpy数组，以便使用seaborn的heatmap函数
components_array = components_df.values

# 使用seaborn的heatmap函数绘制热图，它会处理颜色映射和单元格的数值显示
plt.figure(figsize=(10, 8))  # 设置图形大小
sns.heatmap(components_df, annot=True, fmt=".2f", cmap='viridis', cbar=True, square=True)

# 设置图形标题和轴标签
plt.title('载荷矩阵热图')
plt.xlabel('主成分')
plt.ylabel('变量')

# 显示图形
plt.show()

# ----------------------------保存数据----------------------
# 将PCA结果保存到Excel文件
output_file = 'PCA结果.xlsx'
X_pca_df.to_excel(output_file, index=False)

print(f"PCA results have been saved to {output_file}")

# 将载荷矩阵保存到另一个Excel文件
components_results_file = '载荷矩阵.xlsx'
components_df.to_excel(components_results_file, index=True)
print(f"PCA components have been saved to {components_results_file}")


# 如果需要查看主成分解释的方差比例
print("解释方差比例:", pca.explained_variance_ratio_)

# ------------------------------绘制山崖落石图-----------------
# 使用自助法评估PCA的主成分解释方差比率的分布
np.random.seed(0)
var = []
for i in range(500):
    sample_n = X_scaled[np.random.randint(0, len(X_scaled), len(X_scaled))]
    pca_sample = PCA(n_components=14)
    pca_sample.fit(sample_n)
    var.append(pca_sample.explained_variance_ratio_)

var = np.array(var)

# 绘制每个主成分的解释方差比率的平均值和标准差
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.errorbar(np.linspace(1, 14, 14), np.mean(var, axis=0), yerr=np.std(var, axis=0),
             fmt='o-', capsize=5, elinewidth=1.5, color='b', label='Mean ± STD')

plt.title('主成分解释方差比率分布(碎石图)')  # 设置图形标题
plt.xlabel('主成分编号')  # 设置x轴标签
plt.ylabel('解释方差比率')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()


# ---------------------------------双标图（选择前两个主成分定义的二维平面）--------------------------
def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    # 求所有点的均值作为置信圆的圆心
    pos = points.mean(axis=0)
    # 求协方差
    cov = np.cov(points, rowvar=False)

    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip


# 定义颜色和类别标签，根据您的数据集进行调整
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 示例颜色
# 直接使用df的列名（除去最后一列）作为类别标签
category_labels = df.columns[:-1]

# 确保y是从最后一列提取的类别标签
y = df.iloc[:, -1]


# 修改show_ellipse函数以适应您的数据集
def show_biplot(X_pca, y, pca, feature_label=None):
    plt.figure(dpi=100, figsize=(10, 8))
    xs = X_pca[:, 0]
    ys = X_pca[:, 1]
    unique_categories = np.unique(y)

    # 使用颜色循环为每个类别生成颜色
    color_cycle = plt.get_cmap('hsv', len(unique_categories))

    # 绘制散点图
    for i, category in enumerate(unique_categories):
        mask = y == category
        color = color_cycle(i)  # 获取颜色
        plt.scatter(xs[mask], ys[mask], label=category, color=color, alpha=0.7)

    # 绘制置信椭圆
    for i, category in enumerate(np.unique(y)):
        points = X_pca[y == category]
        plot_point_cov(points, nstd=2, alpha=0.5, color=colors[i])

    # 绘制主成分方向的箭头
    for i in range(pca.components_.shape[0]):
        plt.arrow(0, 0, *pca.components_[i, :2], color='r', alpha=0.5,
                  head_width=0.04, head_length=0.05, overhang=1)
        if feature_label is None:
            plt.text(pca.components_[i, 0] * 1.2, pca.components_[i, 1] * 1.2,
                     f"PC{i + 1}", color='black', ha='center', va='center')
        else:
            plt.text(pca.components_[i, 0] * 1.2, pca.components_[i, 1] * 1.2,
                     feature_label[i], color='black', ha='center', va='center')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
    plt.legend()
    plt.title('PCA Biplot')
    plt.grid(True)
    plt.show()


# 如果需要查看组件的方差
print("组件的方差:", pca.noise_variance_)

show_biplot(X_pca, y, pca, feature_label=df.columns[:-1])
