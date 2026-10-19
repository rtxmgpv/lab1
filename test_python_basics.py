import unittest
from grader_contracts.python_basics import PositiveIntegerInput as N
from grader_contracts.python_basics import TextInput as T
from grader_contracts.python_basics import VectorPairInput as V
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number,
    mse, multiplicative_persistence, prime_factorization, pyramid,
)


class TestCountVowels(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(count_vowels(T("hello")), 2)

    def test_case_insensitive(self):
        self.assertEqual(count_vowels(T("AEIOUaeiou")), 10)

    def test_empty(self):
        self.assertEqual(count_vowels(T("")), 0)

    def test_no_vowels(self):
        self.assertEqual(count_vowels(T("rhythm")), 0)


class TestUniqueCharacters(unittest.TestCase):
    def test_unique(self):
        self.assertTrue(has_unique_characters(T("abcdef")))

    def test_repeat(self):
        self.assertFalse(has_unique_characters(T("abca")))

    def test_empty_and_single(self):
        self.assertTrue(has_unique_characters(T("")))
        self.assertTrue(has_unique_characters(T("a")))

    def test_case_sensitive(self):
        self.assertTrue(has_unique_characters(T("aA")))


class TestOneBits(unittest.TestCase):
    def test_values(self):
        self.assertEqual(count_one_bits(N(1)), 1)
        self.assertEqual(count_one_bits(N(7)), 3)
        self.assertEqual(count_one_bits(N(8)), 1)
        self.assertEqual(count_one_bits(N(255)), 8)

    def test_large(self):
        self.assertEqual(count_one_bits(N(2 ** 100 - 1)), 100)


class TestPersistence(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(multiplicative_persistence(N(39)), 3)
        self.assertEqual(multiplicative_persistence(N(4)), 0)
        self.assertEqual(multiplicative_persistence(N(999)), 4)

    def test_zero_digit(self):
        self.assertEqual(multiplicative_persistence(N(10)), 1)

    def test_known_max(self):
        self.assertEqual(multiplicative_persistence(N(277777788888899)), 11)


class TestMse(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(mse(V([1, 2, 3], [1, 2, 3])), 0.0)
        self.assertAlmostEqual(mse(V([1, 2, 3], [2, 4, 6])), (1 + 4 + 9) / 3)

    def test_single(self):
        self.assertAlmostEqual(mse(V([5], [2])), 9.0)

    def test_negative(self):
        self.assertAlmostEqual(mse(V([-1, -2], [1, 2])), (4 + 16) / 2)

    def test_errors(self):
        with self.assertRaises(ValueError):
            mse(V([1], [1, 2]))
        with self.assertRaises(ValueError):
            mse(V([], []))


class TestPrimeFactorization(unittest.TestCase):
    def test_example(self):
        self.assertEqual(prime_factorization(N(86240)), "(2**5)(5)(7**2)(11)")

    def test_one(self):
        self.assertEqual(prime_factorization(N(1)), "")

    def test_prime(self):
        self.assertEqual(prime_factorization(N(17)), "(17)")

    def test_prime_power(self):
        self.assertEqual(prime_factorization(N(1024)), "(2**10)")

    def test_large_prime_factor(self):
        self.assertEqual(prime_factorization(N(2 * 999983)), "(2)(999983)")


class TestPyramid(unittest.TestCase):
    def test_possible(self):
        self.assertEqual(pyramid(N(1)), 1)
        self.assertEqual(pyramid(N(5)), 2)
        self.assertEqual(pyramid(N(14)), 3)
        self.assertEqual(pyramid(N(55)), 5)

    def test_impossible(self):
        self.assertEqual(pyramid(N(2)), "It is impossible")
        self.assertEqual(pyramid(N(4)), "It is impossible")
        self.assertEqual(pyramid(N(15)), "It is impossible")


class TestBalanced(unittest.TestCase):
    def test_examples(self):
        self.assertTrue(is_balanced_number(N(1234006)))
        self.assertFalse(is_balanced_number(N(123456)))

    def test_short(self):
        self.assertTrue(is_balanced_number(N(7)))
        self.assertTrue(is_balanced_number(N(12)))
        self.assertTrue(is_balanced_number(N(121)))
        self.assertFalse(is_balanced_number(N(132)))

    def test_even(self):
        self.assertTrue(is_balanced_number(N(23441)))
        self.assertTrue(is_balanced_number(N(1331)))

    def test_even_middle_ignored(self):
        self.assertTrue(is_balanced_number(N(1291)))
        self.assertFalse(is_balanced_number(N(1292)))


if __name__ == "__main__":
    unittest.main()
