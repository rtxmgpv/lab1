import unittest
import matplotlib
import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput,
    MatrixInput, MatrixVectorBatchInput, OneHotInput,
    RandomMatrixInput, RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, plot_matrix_histograms, sum_prod,
    unique_columns, unique_rows,
)


matplotlib.use("Agg")


class TestSumProd(unittest.TestCase):
    def test_basic(self):
        matrices = np.array([np.eye(2), 2 * np.eye(2)])
        vectors = np.array([[[1], [2]], [[3], [4]]])
        result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
        np.testing.assert_allclose(result, [[7], [10]])
        self.assertEqual(result.shape, (2, 1))

    def test_single_pair(self):
        a = np.array([[1, 2], [3, 4]])
        x = np.array([[1], [1]])
        result = sum_prod(MatrixVectorBatchInput(a[None], x[None]))
        np.testing.assert_allclose(result, a @ x)

    def test_random_vs_loop(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(4, 3, 3))
        x = rng.normal(size=(4, 3, 1))
        expected = sum(a[i] @ x[i] for i in range(4))
        np.testing.assert_allclose(
            sum_prod(MatrixVectorBatchInput(a, x)), expected)


class TestBinarize(unittest.TestCase):
    def test_strict_threshold(self):
        m = np.array([[0.2, 0.5], [0.7, 1.0]])
        np.testing.assert_array_equal(
            binarize(BinarizeInput(m, 0.5)), [[0, 0], [1, 1]])

    def test_does_not_mutate(self):
        m = np.array([[0.2, 0.9]])
        copy = m.copy()
        binarize(BinarizeInput(m, 0.5))
        np.testing.assert_array_equal(m, copy)

    def test_negative_threshold(self):
        m = np.array([[-1, -3]])
        np.testing.assert_array_equal(binarize(BinarizeInput(m, -2)), [[1, 0]])


class TestUnique(unittest.TestCase):
    def test_rows(self):
        m = np.array([[1, 1, 2], [3, 3, 3]])
        self.assertEqual(unique_rows(MatrixInput(m)), [[1, 2], [3]])

    def test_columns(self):
        m = np.array([[1, 3], [1, 3], [2, 4]])
        self.assertEqual(unique_columns(MatrixInput(m)), [[1, 2], [3, 4]])

    def test_single_element(self):
        m = np.array([[5]])
        self.assertEqual(unique_rows(MatrixInput(m)), [[5]])
        self.assertEqual(unique_columns(MatrixInput(m)), [[5]])

    def test_non_square(self):
        m = np.array([[1, 2, 2, 1]])
        self.assertEqual(unique_rows(MatrixInput(m)), [[1, 2]])
        self.assertEqual(unique_columns(MatrixInput(m)), [[1], [2], [2], [1]])


class TestMatrixStatistics(unittest.TestCase):
    def test_shapes_and_values(self):
        s = matrix_statistics(
            RandomMatrixInput(5, 7, mean=2.0, std=3.0, seed=1))
        self.assertEqual(s.matrix.shape, (5, 7))
        self.assertEqual(s.row_means.shape, (5,))
        self.assertEqual(s.column_means.shape, (7,))
        np.testing.assert_allclose(s.row_means, s.matrix.mean(axis=1))
        np.testing.assert_allclose(s.column_variances, s.matrix.var(axis=0))

    def test_seed_reproducible(self):
        a = matrix_statistics(RandomMatrixInput(3, 3, seed=7))
        b = matrix_statistics(RandomMatrixInput(3, 3, seed=7))
        c = matrix_statistics(RandomMatrixInput(3, 3, seed=8))
        np.testing.assert_array_equal(a.matrix, b.matrix)
        self.assertFalse(np.array_equal(a.matrix, c.matrix))

    def test_distribution(self):
        s = matrix_statistics(
            RandomMatrixInput(200, 200, mean=5.0, std=2.0, seed=0))
        self.assertAlmostEqual(s.matrix.mean(), 5.0, delta=0.1)
        self.assertAlmostEqual(s.matrix.std(), 2.0, delta=0.1)

    def test_plot(self):
        s = matrix_statistics(RandomMatrixInput(4, 6))
        fig = plot_matrix_histograms(s, show=False)
        self.assertEqual(len(fig.axes), 2)


class TestChess(unittest.TestCase):
    def test_example(self):
        result = chess(ChessInput(3, 4, 0, 1))
        np.testing.assert_array_equal(
            result, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    def test_single_cell(self):
        np.testing.assert_array_equal(chess(ChessInput(1, 1, 7, 9)), [[7]])

    def test_single_row_column(self):
        np.testing.assert_array_equal(chess(
            ChessInput(1, 3, 1, 2)), [[1, 2, 1]])
        np.testing.assert_array_equal(
            chess(ChessInput(3, 1, 1, 2)), [[1], [2], [1]])

    def test_floats(self):
        np.testing.assert_allclose(
            chess(ChessInput(2, 2, 0.5, 1.5)), [[0.5, 1.5], [1.5, 0.5]])


RED, BLACK = (255, 0, 0), (0, 0, 0)


class TestRectangle(unittest.TestCase):
    def test_shape_and_center(self):
        img = draw_rectangle(RectangleInput(2, 2, 6, 8, RED, BLACK))
        self.assertEqual(img.shape, (6, 8, 3))
        self.assertEqual(tuple(img[3, 4]), RED)
        self.assertEqual(tuple(img[0, 0]), BLACK)
        self.assertEqual(int((img == RED).all(axis=2).sum()), 4)

    def test_dimensions(self):
        img = draw_rectangle(RectangleInput(4, 2, 6, 8, RED, BLACK))
        mask = (img == RED).all(axis=2)
        self.assertEqual(mask.sum(), 8)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(cols), 4)

    def test_full_image(self):
        img = draw_rectangle(RectangleInput(5, 4, 4, 5, RED, BLACK))
        self.assertTrue((img == RED).all())

    def test_larger_than_image_is_clipped(self):
        img = draw_rectangle(RectangleInput(10, 10, 4, 4, RED, BLACK))
        self.assertEqual(img.shape, (4, 4, 3))
        self.assertTrue((img == RED).all())

    def test_zero_size(self):
        img = draw_rectangle(RectangleInput(0, 0, 4, 4, RED, BLACK))
        self.assertTrue((img == BLACK).all())


class TestEllipse(unittest.TestCase):
    def test_shape_and_center(self):
        img = draw_ellipse(EllipseInput(3, 2, 9, 11, RED, BLACK))
        self.assertEqual(img.shape, (9, 11, 3))
        self.assertEqual(tuple(img[4, 5]), RED)
        self.assertEqual(tuple(img[0, 0]), BLACK)

    def test_symmetry(self):
        img = draw_ellipse(EllipseInput(4, 3, 9, 11, RED, BLACK))
        mask = (img == RED).all(axis=2)
        np.testing.assert_array_equal(mask, mask[::-1])
        np.testing.assert_array_equal(mask, mask[:, ::-1])

    def test_circle_extent(self):
        img = draw_ellipse(EllipseInput(2, 2, 7, 7, RED, BLACK))
        mask = (img == RED).all(axis=2)
        self.assertTrue(mask[3, 1] and mask[3, 5] and mask[1, 3] and mask[5, 3])
        self.assertFalse(mask[3, 0])

    def test_zero_axis(self):
        img = draw_ellipse(EllipseInput(0, 2, 5, 5, RED, BLACK))
        self.assertTrue((img == BLACK).all())

    def test_huge_ellipse_fills_image(self):
        img = draw_ellipse(EllipseInput(100, 100, 5, 5, RED, BLACK))
        self.assertTrue((img == RED).all())


class TestTimeSeries(unittest.TestCase):
    def test_basic(self):
        r = analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 1, 5], 2))
        self.assertAlmostEqual(r.mean, 16 / 6)
        self.assertAlmostEqual(r.variance, np.var([1, 3, 2, 4, 1, 5]))
        self.assertAlmostEqual(r.std, np.std([1, 3, 2, 4, 1, 5]))
        np.testing.assert_array_equal(r.local_maxima_indices, [1, 3])
        np.testing.assert_array_equal(r.local_minima_indices, [2, 4])
        np.testing.assert_allclose(r.moving_average, [2, 2.5, 3, 2.5, 3])

    def test_plateau_not_extremum(self):
        r = analyze_time_series(TimeSeriesInput([1, 2, 2, 1], 1))
        self.assertEqual(len(r.local_maxima_indices), 0)
        self.assertEqual(len(r.local_minima_indices), 0)

    def test_ends_ignored(self):
        r = analyze_time_series(TimeSeriesInput([5, 1, 5], 1))
        np.testing.assert_array_equal(r.local_minima_indices, [1])
        self.assertEqual(len(r.local_maxima_indices), 0)

    def test_short_series(self):
        r = analyze_time_series(TimeSeriesInput([1, 2], 2))
        self.assertEqual(len(r.local_maxima_indices), 0)
        np.testing.assert_allclose(r.moving_average, [1.5])

    def test_window_lengths(self):
        r = analyze_time_series(TimeSeriesInput(list(range(10)), 3))
        self.assertEqual(len(r.moving_average), 8)
        r = analyze_time_series(TimeSeriesInput(list(range(10)), 10))
        np.testing.assert_allclose(r.moving_average, [4.5])

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            analyze_time_series(TimeSeriesInput([1, 2, 3], 4))
        with self.assertRaises(ValueError):
            analyze_time_series(TimeSeriesInput([1, 2, 3], 0))


class TestOneHot(unittest.TestCase):
    def test_example(self):
        r = one_hot(OneHotInput([0, 2, 3, 0]))
        np.testing.assert_array_equal(
            r, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
        )

    def test_explicit_class_count(self):
        r = one_hot(OneHotInput([0, 1], class_count=4))
        self.assertEqual(r.shape, (2, 4))
        np.testing.assert_array_equal(r, [[1, 0, 0, 0], [0, 1, 0, 0]])

    def test_single_label(self):
        np.testing.assert_array_equal(one_hot(OneHotInput([0])), [[1]])

    def test_empty(self):
        self.assertEqual(one_hot(OneHotInput([], class_count=3)).shape, (0, 3))

    def test_errors(self):
        with self.assertRaises(ValueError):
            one_hot(OneHotInput([0, 5], class_count=3))
        with self.assertRaises(ValueError):
            one_hot(OneHotInput([-1, 0]))


if __name__ == "__main__":
    unittest.main()