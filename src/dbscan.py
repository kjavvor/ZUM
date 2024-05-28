from enum import Enum

import numpy as np
import pandas as pd
from scipy.spatial import distance, minkowski_distance
from typing import Union
from sklearn.neighbors import KDTree


class PointState(Enum):
    UNASSIGNED = 0
    NOISE = -1


class DBSCAN:
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> None:
        """Initialize DBSCAN with specified epsilon and minimum samples.

        Parameters:
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.X = None
        self.metric = None
        self.tree = None

        self.metrics = {
            "euclidean": distance.euclidean,
            "minkowski": minkowski_distance,
        }

    def _get_neighbors_(self, point_idx: int) -> np.ndarray:
        """Get the indices of points in X that are neighbors of the point at the given index.

        Parameters:
        point_idx (int): The index of the point to get neighbors for.

        Returns:
        np.ndarray: An array of indices of neighbor points.
        """
        neighbors = []
        for i, point in enumerate(self.X):
            dist = self.metric(self.X[point_idx], point)
            if dist <= self.eps:
                neighbors.append(i)
        return np.array(neighbors)

    # TODO: musimy sie dowiedziec czy mozemy uzywac KDTree z sklearn
    def _get_neighbors(self, point_idx: int) -> np.ndarray:
        """Get the indices of points in X that are neighbors of the point at the given index.

        Parameters:
        point_idx (int): The index of the point to get neighbors for.

        Returns:
        np.ndarray: An array of indices of neighbor points.
        """
        point = self.X[point_idx]
        indices = self.tree.query_radius([point], r=self.eps, return_distance=False)
        return indices[0]

    def _set_metric(self, metric: str) -> None:
        """Set the distance metric for the DBSCAN algorithm.

        Parameters:
            metric (str): The name of the distance metric to use. Must be one of the keys in self.metrics.

        Raises:
            ValueError: If the specified metric is not supported.
        """
        if metric not in self.metrics:
            raise ValueError(
                f"Metric {metric} not supported, must be one of {list(self.metrics.keys())}"
            )
        self.metric = self.metrics[metric]

    def fit_predict(
        self, X: Union[np.ndarray, pd.DataFrame], metric: str = "minkowski"
    ) -> np.ndarray:
        """Perform DBSCAN clustering from features or distance matrix, and return cluster labels.

        Parameters:
        X (np.ndarray): Input data. Array or matrix where each row is a single point.
        metric (str): The name of the distance metric to use. Must be one of the keys in self.metrics.

        Returns:
        np.ndarray: Cluster labels for each point in X. Noisy samples are given the label -1.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.X = X
        self._set_metric(metric)
        self.tree = KDTree(self.X)

        n_points = X.shape[0]
        labels = [PointState.UNASSIGNED] * n_points
        cluster_counter = 0

        for point_idx in range(n_points):
            if labels[point_idx] != PointState.UNASSIGNED:
                continue
            neighbors = self._get_neighbors(point_idx)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = PointState.NOISE

            else:
                queue = neighbors.tolist()

                while queue:
                    current_point = queue.pop(0)

                    if labels[current_point] == PointState.NOISE:
                        labels[current_point] = cluster_counter

                    if labels[current_point] != PointState.UNASSIGNED:
                        continue

                    labels[current_point] = cluster_counter
                    new_neighbors = self._get_neighbors(current_point)

                    if len(new_neighbors) >= self.min_samples:
                        queue += new_neighbors.tolist()

                cluster_counter += 1

        labels = [
            label.value if isinstance(label, PointState) else label for label in labels
        ]

        labels = np.array(labels)

        return labels
