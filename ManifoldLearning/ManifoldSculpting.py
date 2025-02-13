import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from sklearn.decomposition import PCA

class ManifoldSculpting:
    def __init__(self, k_neighbors=15, target_dim=2, iterations=1200, scale_factor=0.99, error_threshold=8e-2, use_pca=True, patience=200):
        """
        Initialize the Manifold Sculpting algorithm.

        Parameters:
        - neighbors (int): Number of nearest neighbors to consider.
        - target_dim (int): Number of dimensions to reduce the dataset to.
        - iterations (int): Maximum number of iterations for optimization.
        - scale_factor (float): Scaling factor for gradual transformation.
        - error_threshold (float): Threshold for convergence.
        - use_pca (bool): Whether to apply PCA preprocessing.
        - patience (int): Number of iterations to wait for improvement before stopping.
        """
        self.k_neighbors = k_neighbors
        self.target_dim = target_dim
        self.iterations = iterations
        self.scale_factor = scale_factor
        self.use_pca = use_pca
        self.error_threshold = error_threshold
        self.patience = patience
        self.error_history = []
        self.error_values = []  # ✅ Store error reduction per iteration
        self.best_representation = None  # ✅ Initialize empty best representation


    def fit_transform(self, data):
        """
        Applies Manifold Sculpting transformation to reduce dimensions.

        Parameters:
        - data (np.array): High-dimensional dataset.

        Returns:
        - np.array: Transformed dataset with reduced dimensions.
       """
        self.scaling_progress = 1
        self.data = data.copy()

        # Compute neighbors, distances, and angle relationships ✅ FIX HERE
        self.neighbors, self.distances_orig, self.avg_distance, self.colinear_pts, self.angles_orig = self._find_neighbors()

        self.learning_rate = self.avg_distance

        # Apply PCA for preprocessing (optional)
        if self.use_pca:
            self.processed_data = self._apply_pca()
            self.d_preserved = np.arange(self.target_dim, dtype=np.int32)
            self.d_scaled = np.arange(self.target_dim, self.data.shape[1], dtype=np.int32)
        else:
            cov_matrix = np.cov(self.data.T)
            sorted_indices = np.argsort(-np.diag(cov_matrix)).astype(np.int32)
            self.d_preserved = sorted_indices[:self.target_dim]
            self.d_scaled = sorted_indices[self.target_dim:]
            self.processed_data = self.data.copy()

        # ✅ Initialize best_representation before optimization
        self.best_representation = self.processed_data.copy()

        # Initial adjustment phase
        epoch = 1
        with tqdm(total=np.inf, desc="Initial Scaling Adjustment") as pbar:
            while self.scaling_progress > 0.01:
                _ = self._adjust_step()
                epoch += 1
                pbar.update(1)

        best_error = np.inf
        no_improvement_epochs = 0

        # Main optimization loop
        with tqdm(total=self.iterations, desc="Optimizing Manifold Sculpting") as pbar:
            pbar.update(epoch)
            while epoch < self.iterations and best_error > self.error_threshold and no_improvement_epochs < self.patience:
                mean_error = self._adjust_step()
                self.error_history.append(mean_error)

                if mean_error < best_error:
                    best_error = mean_error
                    self.best_representation = self.processed_data.copy()
                    self.lowest_error = best_error
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                    if no_improvement_epochs == self.patience:
                        print("Stopping early due to lack of improvement.")
                        break

                epoch += 1
                pbar.update(1)

        self.total_epochs = epoch
        return self.best_representation


    def _find_neighbors(self):
        """
        Computes k-nearest neighbors, distances, and colinear angles.

        Returns:
        - Tuple containing neighbors, distances, avg_distance, colinear points, and angles.
        """
        num_samples = self.data.shape[0]
        knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        knn.fit(self.data)
        distances, indices = knn.kneighbors(self.data)
        neighbor_dists, neighbor_indices = distances[:, 1:], indices[:, 1:]

        avg_distance = np.mean(neighbor_dists)
        colinear_pts = np.zeros((num_samples, self.k_neighbors), dtype=np.int32)
        angles = np.zeros((num_samples, self.k_neighbors), dtype=np.float32)

        for i in range(num_samples):
            for j, neighbor in enumerate(neighbor_indices[i]):
                v1 = self.data[i] - self.data[neighbor]
                norm_v1 = np.linalg.norm(v1)

                neighbor_angles = np.zeros(self.k_neighbors)
                for k, colinear in enumerate(neighbor_indices[neighbor]):
                    v2 = self.data[colinear] - self.data[neighbor]
                    norm_v2 = np.linalg.norm(v2)
                    neighbor_angles[k] = np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1, 1))

                best_colinear_idx = np.argmin(np.abs(neighbor_angles - np.pi))
                colinear_pts[i, j] = neighbor_indices[neighbor, best_colinear_idx]
                angles[i, j] = neighbor_angles[best_colinear_idx]

        return neighbor_indices, neighbor_dists, avg_distance, colinear_pts, angles
    
    def _compute_error(self, point_idx, visited_nodes):
        """
        Computes the transformation error for a given point.

        Parameters:
        - point_idx (int): Index of the data point.
        - visited_nodes (list): List of already adjusted points.

        Returns:
        - float: Total error for the given point.
       """
        weights = np.where(np.isin(self.neighbors[point_idx], visited_nodes), 10, 1)
        # Compute distance error
        a = self.processed_data[point_idx] - self.processed_data[self.neighbors[point_idx]]
        b = self.processed_data[self.colinear_pts[point_idx]] - self.processed_data[self.neighbors[point_idx]]

        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)

        cos_theta = np.clip(np.sum(a * b, axis=1) / (la * lb), -1, 1)
        theta = np.arccos(cos_theta)

        # Compute error components
        err_dist = 0.5 * (la - self.distances_orig[point_idx]) / self.avg_distance
        err_theta = (theta - self.angles_orig[point_idx]) / np.pi

        # Compute total weighted error
        total_error = np.sum(weights * (err_dist**2 + err_theta**2))

        return total_error




    def _apply_pca(self):
        """
        Aligns dataset using PCA to improve convergence.
        Returns transformed data.
        """
        pca = PCA(n_components=self.data.shape[1])
        return pca.fit_transform(self.data)

    def _adjust_step(self):
        """
        Executes one step of the transformation process.

        Returns:
        - mean_error (float): Average transformation error.
        """
        start_idx = np.random.randint(self.data.shape[0])
        queue = [start_idx]
        visited_nodes = []

        self.scaling_progress *= self.scale_factor
        self.processed_data[:, self.d_scaled] *= self.scale_factor

        while self._compute_avg_distance() < self.avg_distance:
            self.processed_data[:, self.d_preserved] /= self.scale_factor

        step_count = 0
        total_error = 0
        processed_count = 0

        while queue:
            idx = queue.pop(0)
            if idx in visited_nodes:
                continue
            queue.extend(self.neighbors[idx])
            s, err = self._adjust_single_point(idx, visited_nodes)
            step_count += s
            total_error += err
            processed_count += 1
            visited_nodes.append(idx)
        
        mean_error=total_error / processed_count
        self.error_values.append(mean_error)  # ✅ Track error values
        return mean_error

    def _compute_avg_distance(self):
        """
        Computes the average neighbor distance.

        Returns:
        - float: Average distance.
        """
        diffs = self.processed_data[:, np.newaxis, :] - self.processed_data[self.neighbors]
        distances = np.linalg.norm(diffs, axis=2)
        return np.sum(distances) / distances.size

    def _adjust_single_point(self, point_idx, visited_nodes):
        """
        Adjusts a single data point using gradient-based optimization.

        Parameters:
        - point_idx (int): Index of the point to adjust.
        - visited_nodes (list): List of already processed points.

        Returns:
        - (int, float): Number of steps and error magnitude.
        """
        learning_factor = self.learning_rate * np.random.uniform(0.3, 1)
        improved = True
        current_error = self._compute_error(point_idx, visited_nodes)
        step_counter = 0

        while step_counter < 30 and improved:
            step_counter += 1
            improved = False

            for i in self.d_preserved:
                self.processed_data[point_idx, i] += learning_factor
                new_error = self._compute_error(point_idx, visited_nodes)

                if new_error >= current_error:
                    self.processed_data[point_idx, i] -= 2 * learning_factor
                    new_error = self._compute_error(point_idx, visited_nodes)

                if new_error >= current_error:
                    self.processed_data[point_idx, i] += learning_factor
                else:
                    current_error = new_error
                    improved = True

        return step_counter - 1, current_error
