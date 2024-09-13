# Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *


def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    result = dot_product(vector1, vector2)

    assert result == 32, f"Expected 32, but got {result}"


def test_cosine_similarity():
    v1, v2 = np.array([1, 0, 0]), np.array([1, 1, 0])

    result = cosine_similarity(v1, v2)

    expected_result = np.dot(
        v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    assert np.isclose(result, expected_result), f"Expected {
        expected_result}, but got {result}"

    assert np.isclose(result, expected_result), f"Expected {
        expected_result}, but got {result}"


def test_nearest_neighbor():
    target_vector = np.array([1, 0, 0])
    vectors = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
    ])

    result = nearest_neighbor(target_vector, vectors)

    expected_index = 1

    assert result == expected_index, f"Expected index {
        expected_index}, but got {result}"
