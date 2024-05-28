import matplotlib.pyplot as plt
from mile.constants import BIRDVIEW_COLOURS
from mile.data.dataset import CarlaDataset
from mile.visualisation import convert_bev_to_image

# Define the configuration for the dataset
cfg = {
    'DATASET_PATH': 'outputs/2024-05-24/16-17-26/dataset',
    'EVAL': {
        'DATASET_REDUCTION': False,
    },
    'BEV': {
        'RESOLUTION': 0.1,  # This is an example, replace with your actual resolution
        'SIZE': [100, 100],  # This is an example, replace with your actual size
        'OFFSET_FORWARD': 10  # This is an example, replace with your actual offset
    }
}

# Create an instance of the CarlaDataset class
dataset = CarlaDataset(cfg=cfg, mode='train', sequence_length=1)

# Retrieve a batch from the dataset
batch = dataset[0]

# Extract the birdview from the batch
birdview = batch['birdview']

# Convert the birdview to an RGB image using the convert_bev_to_image function
birdview_rgb = convert_bev_to_image(birdview, cfg)

# Plot the birdview
plt.imshow(birdview_rgb)
plt.show()