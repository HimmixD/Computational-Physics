import numpy as np

from exercise_02_1d import (
    direct_convolution,
    dipole_kernel_1d,
    dipole_field_xz,
    fft_convolution,
    next_fast_length,
    optimized_fft_convolution,
)


def test_kernel_has_expected_dipole_values_and_zero_self_term():
    kernel = dipole_kernel_1d(4)
    assert kernel[0] == 0.0
    np.testing.assert_allclose(kernel[1], 1.0 / (2.0 * np.pi))
    np.testing.assert_allclose(kernel[2], 1.0 / (16.0 * np.pi))


def test_fft_matches_direct_sum_for_non_power_of_two_size():
    values = np.array([0.0, 1.0, 0.5, 0.0, 2.0, 0.0, 1.0])
    np.testing.assert_allclose(fft_convolution(values), direct_convolution(values), rtol=1e-12, atol=1e-12)


def test_single_cell_has_no_self_field():
    np.testing.assert_allclose(fft_convolution(np.array([3.0])), np.array([0.0]))


def test_dipole_vector_field_reduces_to_1d_kernel_on_x_axis():
    coordinates = np.array([-2.0, 0.0, 2.0])
    field_x, field_z = dipole_field_xz(coordinates, np.array([0.0]))
    np.testing.assert_allclose(field_x[0, [0, 2]], 1.0 / (2.0 * np.pi * 2.0**3))
    np.testing.assert_allclose(field_z[0, [0, 2]], 0.0, atol=1e-15)
    assert np.isnan(field_x[0, 1])
    assert np.isnan(field_z[0, 1])


def test_optimized_fft_matches_direct_sum_with_reused_spectrum():
    values = np.linspace(0.0, 1.0, 11)
    fft_size = next_fast_length(2 * values.size - 1)
    kernel = np.zeros(fft_size)
    kernel[: values.size] = dipole_kernel_1d(values.size)
    kernel[-(values.size - 1) :] = dipole_kernel_1d(values.size)[:0:-1]
    kernel_spectrum = np.fft.rfft(kernel)
    result = optimized_fft_convolution(values, kernel_spectrum, fft_size)
    np.testing.assert_allclose(result, direct_convolution(values), rtol=1e-12, atol=1e-12)


def test_fast_length_is_valid_for_linear_convolution():
    for size in range(1, 100):
        length = next_fast_length(2 * size - 1)
        assert length >= 2 * size - 1
