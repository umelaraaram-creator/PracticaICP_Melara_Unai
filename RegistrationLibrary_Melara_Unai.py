import numpy as np


def calculate_distances_and_correspondences(target, source, max_correspondance_distance):
    """
    Para cada punto del source, encuentra el punto más cercano del target.
    Solo incluye parejas cuya distancia sea menor que max_correspondance_distance.
    """
    correspondences = []
    distances = []

    for i in range(len(source)):
        diffs = target - source[i]
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist <= max_correspondance_distance:
            correspondences.append([i, min_idx])
            distances.append(min_dist)

    return np.array(distances), np.array(correspondences)


def calculate_best_fit_transform(source, target, correspondances):
    """
    Calcula la transformación rígida óptima (R, t) usando SVD.
    """
    # 1. Extraer puntos con correspondencia
    source_corr = source[correspondances[:, 0]]
    target_corr = target[correspondances[:, 1]]

    # 2. Calcular centroides
    centroid_source = np.mean(source_corr, axis=0)
    centroid_target = np.mean(target_corr, axis=0)

    # 3. Centrar los puntos
    source_centered = source_corr - centroid_source
    target_centered = target_corr - centroid_target

    # 4. Matriz de covarianza
    H = source_centered.T @ target_centered

    # 5. Descomposición en valores singulares
    U, _, Vt = np.linalg.svd(H)

    # 6. Matriz de rotación
    R = Vt.T @ U.T

    # 7. Corrección de reflejo
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 8. Traslación
    t = centroid_target - R @ centroid_source

    # 9. Construir la matriz de transformación homogénea
    dim = source.shape[1]
    transformation = np.eye(dim + 1)
    transformation[:dim, :dim] = R
    transformation[:dim, dim] = t

    return transformation


def transform_points(points, transformation):
    """
    Aplica una transformación a los puntos usando coordenadas homogéneas.
    Funciona tanto para 2D como para 3D.
    """
    dim = points.shape[1]
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = homogeneous @ transformation.T
    return transformed[:, :dim]


def calculate_rmse(distances):
    """
    Calcula el Root Mean Square Error a partir del vector de distancias.
    """
    return np.sqrt(np.mean(distances ** 2))


def _estimate_max_correspondence_distance(target, source):
    """
    Estima un max_correspondence_distance razonable basándose en la
    separación real entre las nubes. Calcula la mediana de las distancias
    al vecino más cercano entre source y target, y aplica un factor de
    seguridad x3 para tolerar el desalineamiento inicial.
    """
    n_samples = min(50, len(source))
    sample_dists = []
    for i in range(n_samples):
        dists = np.linalg.norm(target - source[i], axis=1)
        sample_dists.append(np.min(dists))
    return np.median(sample_dists) * 1.0


def icp(target, source,
        max_correspondance_distance=None,
        max_iterations=50,
        metric_delta_threshold=1e-6):
    """
    Iterative Closest Point para registro rígido de nubes de puntos.
    Si max_correspondance_distance es None, se estima automáticamente
    a partir de la separación entre las nubes.
    """
    if max_correspondance_distance is None:
        max_correspondance_distance = _estimate_max_correspondence_distance(target, source)

    src = source.copy()
    prev_metric = float('inf')
    history = []
    dim = source.shape[1]
    total_transformation = np.eye(dim + 1)

    for i in range(max_iterations):
        # Paso 1: Correspondencias
        distances, correspondances = calculate_distances_and_correspondences(
            target, src, max_correspondance_distance)

        # Seguridad: si hay muy pocas correspondencias, parar
        if len(correspondances) < dim + 1:
            break

        # Paso 2: Transformación óptima
        iteration_transformation = calculate_best_fit_transform(
            src, target, correspondances)

        # Paso 3: Acumular transformación
        total_transformation = iteration_transformation @ total_transformation

        # Paso 4: Aplicar transformación
        src = transform_points(src, iteration_transformation)

        # Paso 5: Calcular RMSE
        metric = calculate_rmse(distances)

        history.append((metric, total_transformation))
        if abs(prev_metric - metric) < metric_delta_threshold:
            break
        prev_metric = metric

    return total_transformation, history