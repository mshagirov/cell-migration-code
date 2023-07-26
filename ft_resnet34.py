import fastai
import torch
from fastai.vision.all import *
from fastai.data.all import *
from pathlib import Path
import skimage.io

# Location of the dataset:
data_path = Path('/mnt/mbi/images/micros/murat/dataDIR/maria_21072023/dataset/')

# Location to save the finetuned model:
model_path = Path('.')

# number of epochs
N_epochs = 30

# Package versions and available devices:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"fastai version : {fastai.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Number of GPUs : {torch.cuda.device_count()}")
print(f"Device         : {device}")

# RNG for frame selection
nprng = np.random.default_rng(123)

def imread2pil_train(fpath):
    '''
    read the middle frame and convert to PIL image.
    
    Image files should contain 3 channels.
    '''
    # use print to test datasets:
    # print(fpath)
    
    im = skimage.io.imread(str(fpath))
    
    # assuming images have 3 channels:
    if im.ndim<4:
        im = im[np.newaxis,...]
    
    # later need to add separate normalization for each channel,
    # now using only 2nd ch:CH1 of CH:0-2
    im = np.uint8( 255 * (im/im[:,:,:,1].max()) )
    
    # Filter
    # integrate CH1 along x-axis and find max inten-y ("nucleus")
    Iy = im[...,1].sum(axis=-1).argmax(axis=1)
    
    # frames w/ cells >10% away from the top/bottom edge
    mask = np.abs(Iy - im.shape[1]/2)/im.shape[1] < 0.4
    # if possible, select frames with nuclei away from the edge (visible)
    im = im[...,1][mask] if mask.any() else im[...,1] # only ch=1
    
    # Is est. avg location of nuclei on the top half (True)
    is_top = np.mean(Iy[mask] if mask.any() else Iy) < im.shape[1]/2 # False--> bottom
    
    # range of y-axis to remove empty region (e.g., if is_top==True, then remove bottom)
    y_ids = [0, min([224, im.shape[1]])] if is_top else [max([im.shape[1]-224, 0]), im.shape[1]]
    im = im[:,y_ids[0]:y_ids[1],:]
    
    if fpath.parents[1].name=="train":
        # select random frame if it's training data:
        frame_id = nprng.integers(im.shape[0])
    else:
        # for val data use middle frame
        frame_id = im.shape[0]//2
    return PILImage.create(im[frame_id])



# Training dataset: BW images (2nd channel)
# Images are resized to 224x224 images (cropped) before feeding into the model
# Default training/validation folder names: "train" / "valid"
cell_dataset = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
                         get_items=get_image_files,
                         splitter=GrandparentSplitter(), # folder names: "train" / "valid"
                         get_x=imread2pil_train,
                         get_y=parent_label,
                         item_tfms = Resize(224)
                        )

# debugging: use `cell_dataset.summary(data_path)`

# init dataset
dls = cell_dataset.dataloaders(data_path)

print('\nDataset sizes: \"train, val\"-->',[len(k) for k in dls.splits],'\n')

# Load pretrained model:
# requires internet connection
learn = vision_learner(dls, resnet34, metrics=error_rate)

print(f"\n- - - - -\nTraining for {N_epochs} epochs:\n- - - - -\n")
learn.fine_tune(N_epochs)

# Save the finetuned model for later use:
learn.save( model_path / f'resnet34_finetune_{N_epochs}_filterempty')
