from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton


def calculate_dc(distances: np.ndarray) -> float:
    upper_tri = distances[np.triu_indices(distances.shape[0], k=1)]
    sorted_distances = np.sort(upper_tri)
    n = len(sorted_distances)

    percent = 2.0
    position = int(np.round(n*percent / 100))
    dc = sorted_distances[position]

    return float(dc)


def gaussian_kernel(d_i_j, dc):
    return np.exp(-np.square(d_i_j/dc))


def calculate_local_density(distances: np.ndarray, dc: float):
    """calculate local density for each sample"""
    n_samples = len(distances)
    rho = np.zeros(n_samples)
    for i in range(n_samples-1):
        for j in range(i+1, n_samples):
            rho[i] += gaussian_kernel(distances[i, j], dc)
            rho[j] += gaussian_kernel(distances[i, j], dc)
    return rho


def calculate_delta(distances: np.ndarray, rho):
    n_samples = len(distances)

    delta = np.zeros(n_samples)
    nearest_neighbor = np.full(n_samples, 0)
    # rho降序排序后的数组的元素在rho中的索引
    index_rho = np.argsort(rho)[::-1]

    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点，找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离，并找最小值
        delta[index] = np.min(distances[index, index_higher_rho])

        # 保存最近邻点的索引
        index_nn = np.argmin(distances[index, index_higher_rho])
        nearest_neighbor[index] = index_higher_rho[index_nn].astype(int)

    delta[index_rho[0]] = np.max(delta)
    return delta, nearest_neighbor


def generate_decision_graph(rho, delta, file_path: Path):
    centroids = list()
    fig, ax = plt.subplots(figsize=(10, 6))

    xy = np.vstack([rho, delta])
    z = gaussian_kde(xy)(xy)
    # sc表示散点图上的所有集合
    ax.scatter(rho, delta, c=z, s=30, cmap='plasma', marker='.')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\delta$')
    ax.set_title('Decision Graph')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(file_path.stem + ' Decision Graph.png', dpi=300)

    def select_callback(click_event, release_event):
        # select_callback函数会在用户完成矩形选择后被调用，然后处理用户的选择行为
        x1, y1 = click_event.xdata, click_event.ydata
        x2, y2 = release_event.xdata, release_event.ydata
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        width, height = x_max-x_min, y_max-y_min

        # 保存选中的点; mask是一个布尔掩码
        mask = (rho >= x_min) & (rho <= x_max) & (delta >= y_min) & (delta <= y_max)
        # 获取落在矩形区域内的点的索引(用矩形框选择的聚类中心的索引)
        selected_indices = np.where(mask == 1)[0]
        centroids.extend(selected_indices)

        # 图上添加一个表示选择区域的矩形; Rectangle接收矩形的起点，宽度，和高度
        rect = plt.Rectangle(xy=(x_min, y_min), width=width, height=height,
                             fill=False, edgecolor='blue', linestyle='--')
        ax.add_patch(rect)

        # 更新图形
        fig.canvas.draw()
        print(f"Selected {len(selected_indices)} points")

    _ = RectangleSelector(ax, select_callback, useblit=True, button=[MouseButton.LEFT, MouseButton.RIGHT],
                          minspanx=0, minspany=0, spancoords='data', interactive=True)

    plt.show()
    return centroids


def assign_points_to_clusters(rho, centroids: list[int], nearest_neighbor):
    n_samples = len(rho)

    # 初始化所有样本呢的簇标签为1
    labels = np.full(n_samples, -1)

    for i, centroid in enumerate(centroids):
        # 给聚类中心分配簇标签
        labels[centroid] = i

    index_rho = np.argsort(rho)[::-1]
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labels[index] == -1:
            # 如果没有被标记过，那么簇标签与距离其最近且密度比其大的点的簇标签相同
            labels[index] = labels[int(nearest_neighbor[index])]

    return labels


def visualize_dp_clustering(data, labels, centroids, file_path: Path):
    # k表示聚类中心的数量
    k = len(centroids)
    colors = sns.color_palette("colorblind", k)
    centroids_data = data[centroids]

    for i in range(k):
        # 依次画出k个簇的样本点
        cluster_indices = np.where(labels == i)[0]
        cluster_data = data[cluster_indices]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=15, color=colors[i], marker='.')

    # 画聚类中心
    plt.scatter(centroids_data[:, 0], centroids_data[:, 1], color='red', marker="o", s=100)

    # 判断是否存在未被分配簇标签的样本(簇标签还是-1)，将这些样本画成黑色
    idx = np.where(labels == -1)[0]
    if list(idx):
        other_data = data[idx]
        plt.scatter(other_data[:, 0], other_data[:, 1], color='black', marker=".", s=30)

    plt.savefig(file_path.stem + ' Clustering Result.png', dpi=300)
    plt.show()


def main():
    file_path = Path('./datasets/D1.csv')

    # 加载数据集
    dataset = pd.read_csv(file_path).to_numpy()

    # 计算距离矩阵
    distances = pairwise_distances(dataset)

    # 计算截断距离
    dc = calculate_dc(distances)

    # 计算局部密度
    rho = calculate_local_density(distances, dc)

    # 计算每个样本的相对距离和最近邻
    delta, nearest_neighbor = calculate_delta(distances, rho)

    # 从决策图中选择密度峰值(聚类中心)
    centroids = generate_decision_graph(rho, delta, file_path)

    # 分配非聚类中心到相应的簇并且返回样本簇标签
    labels = assign_points_to_clusters(rho, centroids, nearest_neighbor)

    # 可视化聚类结果
    visualize_dp_clustering(dataset, labels, centroids, file_path)


if __name__ == "__main__":
    main()
