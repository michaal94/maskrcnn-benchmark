import torch.utils.data as data
import os
import json
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList


'''
Create COCO Detection style dataset from CLEVR-Ref+ generated images
(with bounding boxes and segmentation masks).
Compatible with MaskRCNN Benchmark
'''


class CLEVR_segmentation_test(data.Dataset):
    # Initialise dataset from either original file or preprocessed for detection
    def __init__(self, path, ver=None, split='train', transforms=None):
        assert ver is None or ver == 'CoGenT', "Unknown dataset version"
        if ver is None:
            assert split in ['train', 'val', 'test'], "Unknown split for CLEVR"
        else:
            assert split in ['trainA', 'valA', 'valB', 'testA', 'testB'], "Unknown split for CLEVR CoGenT"

        images_dir = os.path.join(path, 'images', split)

        self.test = False

        if 'test' not in split:
            annotation_file = os.path.join(path, ('scenes/CLEVR_%s_scenes.json' % split))
            assert os.path.isfile(annotation_file), 'Annotation file not found'

            with open(annotation_file) as f:
                annotations = json.load(f)['scenes']

            self.ids = [obj['image_index'] for obj in annotations]
            file_names = [os.path.join(images_dir, obj['image_filename']) for obj in annotations]

            self.ids_to_names = dict(zip(self.ids, file_names))
        else:
            file_names = sorted(os.listdir(images_dir))
            self.file_names = [os.path.join(images_dir, f) for f in file_names]
            self.test = True

        self.transforms = transforms
        # # Just a list of some RGB colors, add more if you have more classes
        # self.color_map = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        #                   for i in range(len(self.prop_to_id) + 1)]

    def __getitem__(self, idx):
        if self.test:
            image = Image.open(self.file_names[idx])
        else:
            image = Image.open(self.ids_to_names[self.ids[idx]])

        size = (self.get_img_size(idx)['width'], self.get_img_size(idx)['height'])
        target = BoxList([[0, 0, 0, 0]], size)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, idx

    def visualise_instance(self, idx):
        image, _, _ = self.__getitem__(idx)
        image.show()

    def get_img_info(self, index):
        img_data = self.get_img_size(index)
        return img_data

    def __len__(self):
        return len(self.ids)

    def get_img_size(self, idx):
        width, height = Image.open(self.ids_to_names[self.ids[idx]]).size
        return {'height': height, 'width': width}
