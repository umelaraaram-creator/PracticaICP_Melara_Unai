import numpy as np
from scipy.spatial import cKDTree

_CORRESPONDENCE_DISTANCE_FACTOR = 1.0
_DEFAULT_MAX_ITERATIONS = 50
_DEFAULT_METRIC_DELTA_THRESHOLD = 1e-6


def calculate_distances_and_correspondences(target, source,
                                            max_correspondance_distance):
    """
    Para cada punto del source, busca su vecino más cercano en el target.
    Solo acepta parejas cuya distancia sea <= max_correspondance_distance.

    Utiliza KDTree para búsqueda eficiente de vecinos más cercanos,
    reduciendo la complejidad de O(N*M) a O(N*log(M)).

    Args:
        target (np.ndarray): Nube de referencia (N x dim).
        source (np.ndarray): Nube a registrar (M x dim).
        max_correspondance_distance (float): Distancia máxima permitida.

    Returns:
        distances (np.ndarray): Distancias de las parejas aceptadas.
        correspondences (np.ndarray): Índices [source_idx, target_idx] (K x 2).
    """
    tree = cKDTree(target)
    nearest_distances, nearest_indices = tree.query(source, k=1)

    mask = nearest_distances <= max_correspondance_distance
    valid_source_indices = np.where(mask)[0]
    valid_target_indices = nearest_indices[mask]
    valid_distances = nearest_distances[mask]

    correspondences = np.column_stack((valid_source_indices, valid_target_indices))

    return valid_distances, correspondences


def calculate_best_fit_transform(source, target, correspondances):
    """
    Calcula la transformación rígida óptima (R, t) mediante SVD que
    minimiza la distancia entre los puntos con correspondencia.

    Args:
        source (np.ndarray): Nube source (M x dim).
        target (np.ndarray): Nube target (N x dim).
        correspondances (np.ndarray): Índices [source_idx, target_idx] (K x 2).

    Returns:
        transformation (np.ndarray): Matriz homogénea ((dim+1) x (dim+1)).
    """
    dim = source.shape[1]

    source_matched = source[correspondances[:, 0]]
    target_matched = target[correspondances[:, 1]]

    centroid_source = np.mean(source_matched, axis=0)
    centroid_target = np.mean(target_matched, axis=0)

    source_centered = source_matched - centroid_source
    target_centered = target_matched - centroid_target

    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_target - R @ centroid_source

    transformation = np.eye(dim + 1)
    transformation[:dim, :dim] = R
    transformation[:dim, dim] = t

    return transformation


def transform_points(points, transformation):
    """
    Aplica una transformación rígida usando coordenadas homogéneas.
    Soporta 2D y 3D.

    Args:
        points (np.ndarray): Puntos a transformar (M x dim).
        transformation (np.ndarray): Matriz homogénea ((dim+1) x (dim+1)).

    Returns:
        transformed (np.ndarray): Puntos transformados (M x dim).
    """
    dim = points.shape[1]
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = homogeneous @ transformation.T
    return transformed[:, :dim]


def calculate_rmse(distances):
    """
    RMSE = sqrt( (1/N) * sum(d_i^2) )

    Args:
        distances (np.ndarray): Vector de distancias entre parejas.

    Returns:
        rmse (float): Error cuadrático medio.
    """
    return np.sqrt(np.mean(distances ** 2))


def icp(target, source,
        max_correspondance_distance=None,
        max_iterations=_DEFAULT_MAX_ITERATIONS,
        metric_delta_threshold=_DEFAULT_METRIC_DELTA_THRESHOLD):
    """
    Iterative Closest Point para registro rígido de nubes de puntos.
    Si max_correspondance_distance es None, se estima automáticamente.

    Criterios de parada:
        - Se alcanza max_iterations.
        - La mejora del RMSE es menor que metric_delta_threshold.

    Args:
        target (np.ndarray): Nube de referencia (N x dim).
        source (np.ndarray): Nube a registrar (M x dim).
        max_correspondance_distance (float|None): Distancia máxima.
        max_iterations (int): Número máximo de iteraciones.
        metric_delta_threshold (float): Mejora mínima del RMSE.

    Returns:
        total_transformation (np.ndarray): Transformación acumulada.
        history (list[tuple]): Lista de (rmse, transformación) por iteración.
    """
    if max_correspondance_distance is None:
        max_correspondance_distance = _estimate_max_correspondence_distance(
            target, source)

    dim = source.shape[1]
    min_correspondences = dim + 1
    src = source.copy()
    prev_metric = float('inf')
    total_transformation = np.eye(dim + 1)
    history = []

    for iteration in range(max_iterations):
        distances, correspondances = calculate_distances_and_correspondences(
            target, src, max_correspondance_distance)

        if len(correspondances) < min_correspondences:
            break

        iteration_transformation = calculate_best_fit_transform(
            src, target, correspondances)

        total_transformation = iteration_transformation @ total_transformation
        src = transform_points(src, iteration_transformation)
        metric = calculate_rmse(distances)

        history.append((metric, total_transformation))

        if abs(prev_metric - metric) < metric_delta_threshold:
            break
        prev_metric = metric

    return total_transformation, history


def _estimate_max_correspondence_distance(target, source):
    """
    Estima un max_correspondence_distance razonable usando la mediana
    de las distancias al vecino más cercano. La mediana es robusta
    frente a outliers.

    Args:
        target (np.ndarray): Nube de referencia (N x dim).
        source (np.ndarray): Nube a registrar (M x dim).

    Returns:
        estimated_distance (float): Distancia máxima estimada.
    """
    tree = cKDTree(target)
    nearest_distances, _ = tree.query(source, k=1)
    return np.median(nearest_distances) * _CORRESPONDENCE_DISTANCE_FACTOR