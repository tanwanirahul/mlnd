import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from skimage import exposure
from skimage.filters import gaussian
import random

def load_train_and_validation_set(base_dir, data_dir):
    '''
    Given the base directory and the data directory, load the train and validation datasets from pickle files.
    '''
    train_set = "train.p"
    valid_set = "valid.p"

    # Open the training and validation pickle files to read the serialized data.
    with open(os.path.join(base_dir, data_dir, train_set), mode='rb') as f:
        train = pickle.load(f)
    with open(os.path.join(base_dir, data_dir, valid_set), mode='rb') as f:
        valid = pickle.load(f)

    return (train, valid)


def load_test_set(base_dir, data_dir):
    '''
    Given the base directory and the data directory, load the test dataset from pickle files.
    '''
    test_set = "test.p"

    # Open the test data pickle file to read the serialized data.
    with open(os.path.join(base_dir, data_dir, test_set), mode='rb') as f:
        test = pickle.load(f)

    return test


def plot_class_distirbution(labels, title=""):
    '''
    Given the labels, plot the class distirbution.
    '''
    plt.figure()
    plt.hist(labels)
    plt.title(title)


def load_label_names():
    '''
    Load the descriptive names for traffic sign labels.
    '''
    import csv
    with open('signnames.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        return [row['SignName'] for row in reader]


def plot_images(images, label_indices, label_names, label_preds=None, 
                smooth=True, grid_rows=4, grid_cols=6, is_gray=False):
    '''
        Given the images, label indices, label names, and optional predictions display 
        the images with relevant metadata.
    '''
    assert len(images) == len(label_indices) == (grid_rows * grid_cols)

    # Create figure with sub-plots and set the figure size.
    fig, axes = plt.subplots(grid_rows, grid_cols)
    fig.set_size_inches(18, 8)

    fig.subplots_adjust(hspace=1, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        if is_gray:
            ax.imshow(images[i, :, :], interpolation=interpolation, cmap="gray")
        else:
            ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
            
        # Find the label name from the index.
        cls_true_name = label_names[label_indices[i]]

        # Display the important information about the image.
        if label_preds is None:
            min_value = images[i].min()
            max_value = images[i].max()
            xlabel = "ID: {0}\nLabel: {1}\nMin: {2}\nMax: {3}".format(
               label_indices[i], cls_true_name[:32], min_value, max_value)
        else:
            # Name of the predicted class.
            cls_pred_name = label_names[label_preds[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


def display_sample_images(images, labels, label_names,predictions=None,**kwargs):
    '''
    Given the set of images and labels, display randomly sampled images.
    '''
    samples = np.random.choice(range(len(images)), 24)
    sample_images = np.array([images[i] for i in samples])
    sample_labels = np.array([labels[i] for i in samples])
    if predictions is not None:
        sample_predictions = np.array([predictions[i] for i in samples])
    else:
        sample_predictions = None
    plot_images(sample_images, sample_labels, label_names,label_preds=sample_predictions, **kwargs)


def to_grayscale(images):
    '''
     Given the images in the RGB format, return the gray scaled images.
    '''
    elements = images.shape[0]
    def _to_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    return np.array([_to_grayscale(images[i][:][:][:]) for i in range(elements)])


def normalize(data):
    '''
    Given the array of images, each with dimension 32 x 32, return the 
    normalized image (values in 0 to 1) with the same dimension.
    '''
    data = np.reshape(data, (-1, (32 * 32)))
    normalized = preprocessing.normalize(data)
    return np.reshape(normalized, (-1, 32, 32))


def one_hot_encode(labels):
    '''
    Given the labels, return the one-hot encoded label vectors
    '''
    valid_labels = np.array(range(43))
    hot_encoder = LabelBinarizer().fit(valid_labels)
    hot_encoded = hot_encoder.transform(labels)
    return hot_encoded


def shuffle(features, labels):
    '''
    Given the input images in class sequential order, 
    shuffle the data to randomize the order.
    '''
    shuffled_features, shuffled_labels = sklearn.utils.shuffle(features, labels)
    return (shuffled_features, shuffled_labels)


def _save_preprocessed_data(config, features, labels, data_type="training"):
    '''
    Given the pre-processed data, save it persist it
    under the appropriate directory
    '''
    dir_path = os.path.join(config["local_cache_path"],
        config["preprocessed_data_dir"])

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    save_path = os.path.join(dir_path, data_type + ".p")

    return pickle.dump((features, labels), open(save_path, 'wb'))
    

def _load_preprocessed_data(config, data_type="training"):
    '''
    Load the preprocessed data for the given data_type from the
    corresponding path.
    '''

    load_path = os.path.join(config["local_cache_path"],
        config["preprocessed_data_dir"], data_type + ".p")

    features, labels = pickle.load(open(load_path, mode='rb'))
    return (features, labels)


def save_preprocessed_training_data(config, features, labels):
    '''
    Persist the given pre-processed training dataset under the
    appropriate directory.
    '''
    return _save_preprocessed_data(config, features, labels, "training")


def save_preprocessed_validation_data(config, features, labels):
    '''
    Persist the given pre-processed validation dataset under the 
    appropriate directory.
    '''
    return _save_preprocessed_data(config, features, labels, "validation")


def save_preprocessed_test_data(config, features, labels):
    '''
    Persist the given pre-processed test dataset under the
    appropriate directory.
    '''
    return _save_preprocessed_data(config, features, labels, "test")


def load_preprocessed_training_data(config):
    '''
    Load the persisted pre-processed training dataset from the
    appropriate directory.
    '''
    return _load_preprocessed_data(config, "training")


def load_preprocessed_validation_data(config):
    '''
    Load the persisted pre-processed validation dataset from the
    appropriate directory.
    '''
    return _load_preprocessed_data(config, "validation")


def load_preprocessed_test_data(config):
    '''
    Load the persisted pre-processed test dataset from the
    appropriate directory.
    '''
    return _load_preprocessed_data(config, "test")


def blur(image):
    '''
    Given the input image, make it blur by applying the
    Guassian filter.
    '''
    rand = random.random()
    return gaussian(image, sigma=rand)


def constrast_stretching(images, is_gray=True):
    '''
    Given the images, return the images after applying
    contrast stretching to each of the elements.
    '''
    dimension = (32, 32) if is_gray else (32, 32, 3)
    image_size = (32 * 32) if is_gray else (32 * 32 * 3)

    def apply_rescale_intensity(img, percentile_range = (3, 97)):
        '''
        Given the image, apply the constrast stretching.
        '''
        img = img.reshape(dimension)
        lb, ub = np.percentile(img, percentile_range)
        img_rescaled = exposure.rescale_intensity(img, in_range=(lb, ub))
        return img_rescaled.reshape(-1, image_size)

    scaled_images = np.apply_along_axis(apply_rescale_intensity, 1, images.reshape(-1, image_size))
    return scaled_images.reshape(-1, *dimension)


def adaptive_histograms(images, is_gray=True, clip_limit=0.1, add_blur=True):
    '''
    Given the images, return the images after applying
    adaptive histogram to each of the elements.
    '''
    dimension = (32, 32) if is_gray else (32, 32, 3)
    image_size = (32 * 32) if is_gray else (32 * 32 * 3)

    def apply_adaptive_hist(img):
        '''
        Given the image, apply the adaptive histogram.
        '''
        img = img.reshape(dimension)
        img_processed = exposure.equalize_adapthist(img, clip_limit=clip_limit)
        if add_blur:
             img_processed = blur(img_processed)
        
        return img_processed.reshape(image_size)

    scaled_images = np.apply_along_axis(apply_adaptive_hist, 1, images.reshape(-1, image_size))
    return scaled_images.reshape(-1, *dimension)


def equalize_hist(images, is_gray=True):
    '''
    Given the images, return the images after applying
    histogram equalization to each of the elements.
    '''
    dimension = (32, 32) if is_gray else (32, 32, 3)
    image_size = (32 * 32) if is_gray else (32 * 32 * 3)

    def apply_equalize_hist(img):
        '''
        Given the image, apply the histogram eqalization transform.
        '''
        img = img.reshape(dimension)
        img_processed = exposure.equalize_hist(img, nbins=512)
        return img_processed.reshape(image_size)

    scaled_images = np.apply_along_axis(apply_equalize_hist, 1, images.reshape(-1, image_size))
    return scaled_images.reshape(-1, *dimension)
