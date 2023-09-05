from u2net_network import U2NET
import os
from PIL import Image
import cv2
import gdown
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))  # Change device to cuda if GPU is available and CUDA is configured
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"



def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)



def generate_mask(input_image_path, net, palette, device, output_folder, cloth_class):

    img = Image.open(input_image_path)
    # img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    # Check which classes are present in the image
    # Change this to range(1, 4) to include all classes - top, bottom, full
    if cloth_class == 'top':
        cls = 1

    elif cloth_class == 'bottom':
        cls = 2

    elif cloth_class == 'full':
        cls = 3
        
    alpha_mask = (output_arr == cls).astype(np.uint8) * 255
    alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
    alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
    alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)

    input_filename = os.path.basename(input_image_path)  # Get the base filename of the input image
    output_filename = input_filename # Replace the extension and add class
    output_path = os.path.join(output_folder, output_filename)  # Combine output folder and filename

    alpha_mask_img.save(output_path)  # Save the alpha mask image with the constructed output path


def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=1tAVTaNDCXSOBK_GOUayKcUoy3q8lLEvA"
        gdown.download(url, file_path, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)   # changed to 32 from 3
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net

def process_images(input_folder, output_folder, cloth_class):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Create an instance of your model
    checkpoint_path = 'model/cloth_segm.pth'
    model = load_seg_model(checkpoint_path, device=device)

    palette = get_palette(4)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')]

    for image_file in image_files:
        image_file = os.path.join(input_folder, image_file)
        
        generate_mask(image_file, model, palette, device, output_folder, cloth_class)


def extract_cloth(input_folder, mask_folder, output_folder):

    # Get a list of all image filenames in the input folder
    input_image_filenames = [filename for filename in os.listdir(input_folder) if filename.endswith('.png')]

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_filename in input_image_filenames:
        input_path = os.path.join(input_folder, input_filename)
        mask_filename = os.path.splitext(input_filename)[0] + '.png'  # Change mask filename
        mask_path = os.path.join(mask_folder, mask_filename)

        if os.path.exists(mask_path):
            original_image = cv2.imread(input_path)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            #convert mask_image to RGBA
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGBA)

            # Iterate through each pixel and set alpha to 0 for black region
            for y in range(mask_image.shape[0]):
                for x in range(mask_image.shape[1]):
                    # Check if the pixel is black (assuming black is [0, 0, 0] in BGR)
                    if all(mask_image[y, x, 0:3] == [0, 0, 0]):
                        mask_image[y, x, 3] = 0  # Set alpha channel to 0 for transparency

            output_filename = os.path.splitext(input_filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, mask_image)
            overlay_images(input_path, output_path, output_path)


def overlay_images(original_path, mask_path, output_path):
    original = Image.open(original_path).convert('RGBA')
    mask = Image.open(mask_path).convert('RGBA')

    # Ensure both images have the same dimensions
    if original.size != mask.size:
        mask = mask.resize(original.size, Image.ANTIALIAS)

    original_data = np.array(original)
    mask_data = np.array(mask)

    # Use NumPy to efficiently apply overlay only on white regions of the mask
    overlay_region = (mask_data[:, :, 0] == 255) & (mask_data[:, :, 1] == 255) & (mask_data[:, :, 2] == 255)
    merged_data = np.where(np.repeat(overlay_region[:, :, np.newaxis], 4, axis=2), original_data, [0, 0, 0, 0])

    merged = Image.fromarray(merged_data.astype(np.uint8))

    output_filename = os.path.splitext(output_path)[0] + '.png'
    merged.save(output_filename, "PNG")


if __name__ == '__main__':

    input_folder = './inputs'  # Change this to your folder that contains png images of fashion items
    output_folder = './imgs_out'  # Change this to your output folder where you want the extracted clothing images to be saved 

    """
    Change cloth_class to 'top', 'bottom' or 'full' depending on the type of clothing you want to extract.
    For example a shirt would belong to the 'top' class, while a pair of jeans would belong to the 'bottom' class.
    Similarly, a full dress like a gown would belong to the 'full' class.
    """
    cloth_class = 'top'

    process_images(input_folder, output_folder, cloth_class)

    extract_cloth(input_folder, output_folder, output_folder) # The final segmented cloth will be saved in the `output folder`