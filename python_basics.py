"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import (
    PositiveIntegerInput, TextInput, VectorPairInput
)
from collections import Counter


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(1 for ch in text.lower() if ch in "aeiou")


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Векторы разной длины!")
    if len(predicted) == 0:
        raise ValueError("Векторы пустые!")
    total = sum((p - e) ** 2 for p, e in zip(predicted, expected))
    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors: Counter[int] = Counter()
    divisor = 2
    while divisor * divisor <= number:
        while number % divisor == 0:
            factors[divisor] += 1
            number //= divisor
        divisor += 1 if divisor == 2 else 2
    if number > 1:
        factors[number] += 1

    parts = []
    for prime in sorted(factors):
        power = factors[prime]
        parts.append(f"({prime})" if power == 1 else f"({prime}**{power})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 0
    remaining = cube_count
    while remaining > 0:
        k += 1
        remaining -= k * k
    return k if remaining == 0 and k > 0 else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(ch) for ch in str(number)]
    half = (len(digits) - 1) // 2
    left = digits[:half]
    right = digits[len(digits) - half:]
    return sum(left) == sum(right)
