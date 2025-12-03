"""
Unit tests for the read_image module.
"""

import sys

sys.path.insert(0, "..")
import pytest
from common import np
from read_image import read_in_image


class TestReadImage:
    """Test suite for read_image module."""

    def test_read_3d_image(self, sample_3d_image):
        """Test reading a 3D TIFF image."""
        img_path, expected_img_data = sample_3d_image

        # Read the image
        img_data, unique_ids, num_pixels = read_in_image(img_path, "fuel cell", 3)

        # Check image data matches
        np.testing.assert_array_equal(img_data, expected_img_data)

        # Check unique grain IDs
        assert set(unique_ids) == {0, 1, 2}  # pore, electrolyte, electrode

        # Check pixel dimensions
        assert len(num_pixels) == 3
        assert num_pixels[0] == 5  # x dimension
        assert num_pixels[1] == 5  # y dimension
        assert num_pixels[2] == 5  # z dimension

    def test_read_2d_image(self, sample_2d_image):
        """Test reading a 2D TIFF image."""
        img_path, expected_img_data = sample_2d_image

        # Read the image
        img_data, unique_ids, num_pixels = read_in_image(img_path, "fuel cell", 2)

        # Check image data matches
        np.testing.assert_array_equal(img_data, expected_img_data)

        # Check unique grain IDs
        assert set(unique_ids) == {0, 1, 2}  # pore, electrolyte, electrode

        # Check pixel dimensions
        assert len(num_pixels) == 2
        assert num_pixels[0] == 10  # x dimension
        assert num_pixels[1] == 10  # y dimension

    def test_unique_grain_ids_extraction(self, sample_3d_image):
        """Test that unique grain IDs are correctly extracted."""
        img_path, _ = sample_3d_image

        # Read the image
        _, unique_ids, _ = read_in_image(img_path, "fuel cell", 3)

        # Check that IDs are sorted
        assert unique_ids == sorted(unique_ids)

        # Check that all IDs are integers
        assert all(isinstance(id, int) for id in unique_ids)

    def test_pixel_count_accuracy(self, sample_3d_image):
        """Test that pixel counts match actual image dimensions."""
        img_path, expected_img_data = sample_3d_image

        # Read the image
        _, _, num_pixels = read_in_image(img_path, "fuel cell", 3)

        # Compare with numpy shape
        assert num_pixels[0] == expected_img_data.shape[0]
        assert num_pixels[1] == expected_img_data.shape[1]
        assert num_pixels[2] == expected_img_data.shape[2]

    def test_fuel_cell_physics_type(self, sample_3d_image):
        """Test reading image for fuel cell physics."""
        img_path, _ = sample_3d_image

        # Should work with "fuel cell" physics type
        img_data, unique_ids, num_pixels = read_in_image(img_path, "fuel cell", 3)

        assert img_data is not None
        assert len(unique_ids) > 0
        assert len(num_pixels) == 3

    def test_battery_physics_type(self, sample_3d_image):
        """Test reading image for battery physics."""
        img_path, _ = sample_3d_image

        # Should also work with "battery" physics type
        img_data, unique_ids, num_pixels = read_in_image(img_path, "battery", 3)

        assert img_data is not None
        assert len(unique_ids) > 0
        assert len(num_pixels) == 3

    def test_invalid_file_path(self):
        """Test handling of invalid file path."""
        with pytest.raises(FileNotFoundError):
            read_in_image("nonexistent_file.tif", "fuel cell", 3)

    def test_dimension_mismatch_2d_vs_3d(self, sample_2d_image, sample_3d_image):
        """Test that dimension parameter correctly affects output."""
        img_2d_path, _ = sample_2d_image
        img_3d_path, _ = sample_3d_image

        # Read 2D image with dimension=2
        _, _, num_pixels_2d = read_in_image(img_2d_path, "fuel cell", 2)
        assert len(num_pixels_2d) == 2

        # Read 3D image with dimension=3
        _, _, num_pixels_3d = read_in_image(img_3d_path, "fuel cell", 3)
        assert len(num_pixels_3d) == 3

    def test_phase_identification(self, temp_dir):
        """Test correct identification of different phases."""
        # Create a custom image with specific phase distribution
        img_data = np.zeros((8, 8, 8), dtype=np.uint8)
        img_data[:3, :, :] = 0  # 192 voxels of pore
        img_data[3:5, :, :] = 1  # 128 voxels of electrolyte
        img_data[5:, :, :] = 2  # 192 voxels of electrode

        import tifffile

        img_path = temp_dir / "phase_test.tif"
        tifffile.imwrite(str(img_path), img_data)

        # Read and verify
        read_img, unique_ids, _ = read_in_image(str(img_path), "fuel cell", 3)

        # Check phases
        assert 0 in unique_ids  # pore
        assert 1 in unique_ids  # electrolyte
        assert 2 in unique_ids  # electrode

        # Verify phase distribution
        assert np.sum(read_img == 0) == 192
        assert np.sum(read_img == 1) == 128
        assert np.sum(read_img == 2) == 192

    def test_single_phase_image(self, temp_dir):
        """Test handling of single-phase image."""
        # Create an image with only one phase
        img_data = np.ones((5, 5, 5), dtype=np.uint8)

        import tifffile

        img_path = temp_dir / "single_phase.tif"
        tifffile.imwrite(str(img_path), img_data)

        # Read the image
        _, unique_ids, _ = read_in_image(str(img_path), "fuel cell", 3)

        # Should only have one unique ID
        assert len(unique_ids) == 1
        assert unique_ids[0] == 1
