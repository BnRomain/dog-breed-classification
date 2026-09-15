"""Unit tests of the functions of src/preprocessing_knn.py (importable thanks to pythonpath = ["src"])."""

import numpy as np
import pytest
from skimage import io
from skimage.filters import sobel_h, sobel_v

import preprocessing_knn as pk

BOX_XML = (
    "<annotation><object><bndbox>"
    "<xmin>20</xmin><ymin>10</ymin><xmax>49</xmax><ymax>29</ymax>"
    "</bndbox></object></annotation>"
)


def make_small_db(root):
    """Creates a tiny dataset with the SmallDB layout: two breeds, three images with a bounding box."""
    for breed, files in (("n001-alpha", ["a1.png", "a2.png"]), ("n002-beta", ["b1.png"])):
        img_dir = root / "Images" / breed
        ann_dir = root / "Annotation" / breed
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        img = np.zeros((40, 60, 3), dtype=np.uint8)
        img[10:30, 20:50] = 255
        for name in files:
            io.imsave(img_dir / name, img, check_contrast=False)
            (ann_dir / name[:-4]).write_text(BOX_XML)


@pytest.fixture
def pca_data():
    rng = np.random.default_rng(1)
    return rng.random((40, 16))


# --- Entropy ---------------------------------------------------------------


def test_entropy_of_balanced_classes_is_log_n():
    assert pk.entropy([10, 10, 10, 10]) == pytest.approx(np.log(4))


def test_entropy_of_a_single_class_is_zero():
    assert pk.entropy([0, 7, 0]) == pytest.approx(0.0)


def test_entropy_normalizes_the_counts():
    assert pk.entropy([1, 3]) == pytest.approx(pk.entropy([0.25, 0.75]))


def test_entropy_breeds_uses_the_class_counts():
    labels = [0] * 152 + [1] * 175 + [2] * 179
    assert pk.entropy_breeds(labels) == pytest.approx(pk.entropy([152, 175, 179]))


# --- Dataset loading --------------------------------------------------------


def test_extract_bounding_box_parses_the_pascal_voc_xml(tmp_path):
    box = tmp_path / "box"
    box.write_text(BOX_XML)
    assert pk.extract_bounding_box(box) == (20, 49, 10, 29)


def test_read_and_crop_db_returns_the_crops_and_the_labels(tmp_path):
    make_small_db(tmp_path)
    images, dogs, labels, label_names = pk.read_and_crop_db(TO_DB=str(tmp_path), color=False)
    assert label_names == {0: "alpha", 1: "beta"}
    assert labels == [0, 0, 1]
    assert images[0].shape == (40, 60)
    assert dogs[0].shape == (20, 30)
    # The crop contains only the white rectangle.
    assert dogs[0].min() == pytest.approx(1.0)


def test_read_and_crop_db_without_dataset_returns_none(tmp_path):
    assert pk.read_and_crop_db(TO_DB=str(tmp_path / "missing")) == (None, None, None, None)


def test_read_db_loads_the_color_images(tmp_path):
    make_small_db(tmp_path)
    images, labels, label_names = pk.read_db(TO_DB=str(tmp_path / "Images"), color=True)
    assert len(images) == 3
    assert images[0].shape == (40, 60, 3)
    assert labels == [0, 0, 1]
    assert label_names == {0: "alpha", 1: "beta"}
    assert pk.read_db(TO_DB=str(tmp_path / "missing")) == (None, None, None)


# --- Resizing and padding ---------------------------------------------------


@pytest.mark.parametrize("shape", [(30, 90), (90, 30), (50, 50)])
def test_resize_and_pad_keeps_the_aspect_ratio(shape):
    img = np.full(shape, 0.5)
    out = pk.resize_and_pad(img, target_size=(64, 64), pad_type="black")
    assert out.shape == (64, 64)
    rows = np.flatnonzero(out.max(axis=1) > 0)
    cols = np.flatnonzero(out.max(axis=0) > 0)
    content_h = rows[-1] - rows[0] + 1
    content_w = cols[-1] - cols[0] + 1
    assert max(content_h, content_w) == 64
    assert content_h / content_w == pytest.approx(shape[0] / shape[1], rel=0.1)


def test_resize_and_pad_fills_the_borders_with_white_or_black():
    img = np.full((20, 60), 0.5)
    white = pk.resize_and_pad(img, (64, 64), "white")
    black = pk.resize_and_pad(img, (64, 64), "black")
    assert white[0, 0] == 1.0 and white[-1, -1] == 1.0
    assert black[0, 0] == 0.0 and black[-1, -1] == 0.0
    assert white[32, 32] == pytest.approx(0.5)
    assert black[32, 32] == pytest.approx(0.5)


@pytest.mark.parametrize("pad_type", ["continuous", "reflect", "propagated_blur"])
def test_resize_and_pad_content_based_padding_stays_in_range(pad_type):
    rng = np.random.default_rng(0)
    img = rng.random((20, 60))
    out = pk.resize_and_pad(img, (64, 64), pad_type)
    assert out.shape == (64, 64)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_and_pad_keeps_the_color_channels():
    img = np.full((30, 90, 3), 0.5)
    for pad_type in ("white", "continuous", "propagated_blur"):
        out = pk.resize_and_pad(img, (64, 64), pad_type)
        assert out.shape == (64, 64, 3)
    continuous = pk.resize_and_pad(img, (64, 64), "continuous")
    assert continuous[0, 0] == pytest.approx([0.5, 0.5, 0.5])


def test_get_resized_db_stacks_the_images():
    imgs = [np.zeros((10, 20)), np.ones((40, 10))]
    out = pk.get_resized_db(imgs, target_size=(32, 32), pad_type="black")
    assert out.shape == (2, 32, 32)
    assert pk.get_resized_db([], target_size=(32, 32)).size == 0


def test_img_rescale_and_img_resize():
    gray = np.ones((20, 40))
    assert pk.img_rescale(gray, 0.5).shape == (10, 20)
    assert pk.img_rescale(np.ones((20, 40, 3)), 0.5).shape == (10, 20, 3)
    assert pk.img_resize(gray, (8, 8)).shape == (8, 8)


def test_convert_ndarrays2data_matrix_flattens_each_image():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    mtx = pk.convert_ndarrays2data_matrix(arr)
    assert mtx.shape == (2, 12)
    assert np.array_equal(mtx[1], np.arange(12, 24))


# --- PCA and HOG features ---------------------------------------------------


def test_project_onto_pca_keeps_the_first_components(pca_data):
    pca = pk.my_PCA(pca_data, n_components=8)
    proj = pk.project_onto_PCA(3, pca, pca_data)
    assert proj.shape == (40, 3)
    assert np.allclose(proj, pca.transform(pca_data)[:, :3])


def test_compute_pca_features_concatenates_the_class_models(pca_data):
    pcas = [pk.my_PCA(pca_data[:20], n_components=5), pk.my_PCA(pca_data[20:], n_components=5)]
    features = pk.compute_pca_features(pcas, pca_data, nb_components=4)
    assert features.shape == (40, 8)


def test_compute_hog_accumulates_the_gradient_magnitudes():
    rng = np.random.default_rng(2)
    img = rng.random((64, 64))
    hog = pk.compute_hog(img, nb_height_cells=4, nb_width_cells=4, nb_bins=8)
    assert hog.shape == (4 * 4 * 8,)
    magnitude = np.hypot(sobel_h(img), sobel_v(img))
    assert hog.sum() == pytest.approx(magnitude.sum())


def test_compute_hog_puts_a_vertical_edge_in_a_single_bin():
    img = np.zeros((16, 16))
    img[:, 8:] = 1.0
    hog = pk.compute_hog(img, nb_height_cells=1, nb_width_cells=1, nb_bins=8)
    assert hog.sum() > 0
    # A vertical edge has a purely horizontal gradient: orientation pi/2, fifth bin.
    assert hog[4] == pytest.approx(hog.sum())


# --- Figures ----------------------------------------------------------------


def test_plot_functions_write_the_figures(tmp_path, pca_data):
    labels = [0] * 20 + [1] * 20
    label_names = {0: "alpha", 1: "beta"}
    pca = pk.my_PCA(pca_data)

    pk.plot_barplot(labels, label_names, fig_path=tmp_path / "balance.png")
    pk.visualize_var_pcs(pca, fig_path=tmp_path / "scree.png")
    pk.plot_whole_db_on_2d(pca, pca_data, labels, fig_path=tmp_path / "scatter.png")
    pk.display_pca_approx(pca_data[0], pca, target_size=(4, 4), fig_path=tmp_path / "approx.png")
    pk.plot_images([np.zeros((5, 5))] * 12, labels[:12], label_names, fig_path=tmp_path / "images.png")

    for name in ("balance.png", "scree.png", "scatter.png", "approx.png", "images.png"):
        assert (tmp_path / name).stat().st_size > 0
