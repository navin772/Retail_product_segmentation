# Retail Fashion

This `retail_fashion` directory contains the necessary files for processing the fashion clothing images and creating their masks.

## File Structure

- The `network.py` contains the [U2-Net](https://github.com/levindabhi/cloth-segmentation) model layers for generating masks of segmented cloth images.

- The `process_extract_overlay.py` file loads the model and first creates the mask of the cloth and then overlays it with the original image to render only the cloth with a transparent background. This helps in creating a better visualization of the cloth and thus better vector embeddings. 

- `model/cloth_segm.pth` is the model that is used for generating the masks. running the `process_extract_overlay.py` file for the first time will download the model automatically to the `model` directory.

## Utilizing the code

- In the `process_extract_overlay.py` file, change the `input_folder` variable to the path of the folder containing the images to be processed. The images should be in the `.png` format. And change the `output_folder` variable to the path of the folder where the processed images should be saved.

- Also, change the `cloth_class` variable in line 236 to the type of the cloth class you are processing. A given folder should contain only one type of cloth class. For example, if you are processing the images of t-shirts, then the folder should contain only t-shirt images.

- Run the `process_extract_overlay.py` file as `python process_extract_overlay.py`. This will create an output folder with containing the processed images.

- Now you can create the vector embeddings from the images in the output folder.

## References, citations and contributions

- [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007) by Qijie Zhao, Ming-Ming Cheng, Shikui Wei, Xiao Liu, Wangmeng Zuo, Jiaying Liu, Zhangyang Wang.
- [Cloth Segmentation](https://github.com/levindabhi/cloth-segmentation) implementation of U2-Net by Levin Dabhi.

- [Akash Kumar](https://github.com/AkashKumar7902) for helping with the code and implementation.