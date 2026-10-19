"""Заготовки задач на NumPy."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    if vectors.ndim == 2:
        vectors = vectors[:, :, np.newaxis]
    return np.matmul(matrices, vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_matrix_histograms(stats: MatrixStatistics, bins: int = 20, show: bool = True):
    import matplotlib.pyplot as plt

    matrix = np.asarray(stats.matrix)
    fig, (ax_rows, ax_cols) = plt.subplots(1, 2, figsize=(12, 4))

    for i, row in enumerate(matrix):
        ax_rows.hist(
            row, bins=bins, alpha=0.4, label=f"строка {i}" if matrix.shape[0] <= 5 else None)
    ax_rows.set_title("Гистограммы значений по строкам")
    ax_rows.set_xlabel("Значение")
    ax_rows.set_ylabel("Частота")

    for j, column in enumerate(matrix.T):
        ax_cols.hist(
            column, bins=bins, alpha=0.4, label=f"столбец {j}" if matrix.shape[1] <= 5 else None)
    ax_cols.set_title("Гистограммы значений по столбцам")
    ax_cols.set_xlabel("Значение")
    ax_cols.set_ylabel("Частота")

    for ax in (ax_rows, ax_cols):
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i, j = np.indices((rows, columns))
    return np.where((i + j) % 2 == 0, first, second)


def _new_image(height: int, width: int, background_color) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = np.asarray(background_color, dtype=np.uint8)
    return image


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = _new_image(image_height, image_width, background_color)

    top = max((image_height - height) // 2, 0)
    left = max((image_width - width) // 2, 0)
    bottom = min(top + height, image_height)
    right = min(left + width, image_width)
    if height > 0 and width > 0:
        image[top:bottom, left:right] = np.asarray(shape_color, dtype=np.uint8)
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = _new_image(image_height, image_width, background_color)

    if semi_axis_x <= 0 or semi_axis_y <= 0:
        return image

    y0 = (image_height - 1) / 2
    x0 = (image_width - 1) / 2
    y, x = np.ogrid[:image_height, :image_width]
    inside = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[inside] = np.asarray(shape_color, dtype=np.uint8)
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("Временной ряд должен быть одномерным!")
    if not 1 <= window <= len(series):
        raise ValueError("Размер окна должен быть от 1 до длины ряда!")

    middle = series[1:-1]
    maxima = np.flatnonzero((middle > series[:-2]) & (middle > series[2:])) + 1
    minima = np.flatnonzero((middle < series[:-2]) & (middle < series[2:])) + 1
    moving_average = sliding_window_view(series, window).mean(axis=1)

    return TimeSeriesStatistics(
        mean=float(series.mean()),
        variance=float(series.var()),
        std=float(series.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if labels.size and labels.min() < 0:
        raise ValueError("Метки классов должны быть неотрицательными!")
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    elif labels.size and labels.max() >= class_count:
        raise ValueError("Метка класса выходит за пределы class_count!")

    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
