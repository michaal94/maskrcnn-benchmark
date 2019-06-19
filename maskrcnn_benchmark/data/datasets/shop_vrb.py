import torch
import torch.utils.data as data
import os
import json
from PIL import Image, ImageDraw
import pycocotools.mask as mask_utils
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import numpy as np


class SHOP_VRB(data.Dataset):
    # Baseline for different tasks
    def __init__(self, path, transforms=None, split='train'):
        assert split in ['train', 'val', 'test', 'benchmark'], 'Split not found'
        self.images_dir = os.path.join(path, 'images', split)
        annotation_file = os.path.join(path, 'scenes', 'SHOP_VRB_' + split + '_scenes.json')
        assert os.path.isfile(annotation_file), 'Annotation file not found'
        objects_to_nums_file = os.path.join(path, 'SHOP_VRB_obj_name_to_num.json')
        assert os.path.isfile(objects_to_nums_file), 'Dictionary file not found'

        with open(objects_to_nums_file) as f:
            self.name_to_id = json.load(f)

        with open(annotation_file) as f:
            annotations = json.load(f)['scenes']

        self.ids = [obj['image_index'] for obj in annotations]
        file_names = [os.path.join(self.images_dir, obj['image_filename']) for obj in annotations]
        self.ids_to_names = dict(zip(self.ids, file_names))
        self.annotations = annotations
        self.transforms = transforms

    def __getitem__(self, idx):
        image = Image.open(self.ids_to_names[self.ids[idx]])
        annotation = self.annotations[idx]
        return image, annotation

    def __len__(self):
        return len(self.ids)

    def get_img_size(self, idx):
        img_size = self.annotations[idx]['objects'][0]['mask']['size']
        # print(img_size)
        return {'height': img_size[0], 'width': img_size[1]}


class SHOP_VRB_mask(SHOP_VRB):
    # Initialise dataset from either original file or preprocessed for detection
    def __init__(self, path, transforms=None, split='train'):
        super(SHOP_VRB_mask, self).__init__(path, transforms=transforms, split=split)
        assert split in ['train', 'val', 'test', 'benchmark']
        preprocessed_file = os.path.join(
            path, 'scenes', 'SHOP_VRB_' + split + '_scenes_maskrcnn.json')
        if not os.path.isfile(preprocessed_file):
            self.preprocess_dataset(preprocessed_file)
        with open(preprocessed_file) as f:
            annotations = json.load(f)

        self.ids = [obj['image_index'] for obj in annotations]
        file_names = [os.path.join(self.images_dir, obj['image_filename']) for obj in annotations]

        self.ids_to_names = dict(zip(self.ids, file_names))
        self.annotations = annotations
        self.transforms = transforms
        # Just a list of some RGB colors, add more if you have more classes
        self.color_map = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                          for i in range(len(self.name_to_id) + 1)]

    # Preprocess dataset only for idxs, filenames, labels, boxes, masks
    def preprocess_dataset(self, path_to_save):
        print("Preprocessed annotation file not found")
        print("Preprocessing dataset for COCO detection format annotations...")
        from tqdm import tqdm
        new_annotations = []
        for idx in tqdm(range(self.__len__())):
            new_annotation = self.annotations[idx]
            img_size = new_annotation['objects'][0]['mask']['size']
            for obj in new_annotation['objects']:
                rle_mask = obj['mask']
                mask = mask_utils.decode(rle_mask)
                bbox = mask_to_bbox(mask, img_size)
                area = mask.sum()
                if not (area > 0 and bbox[2] > bbox[0] and bbox[3] > bbox[1]):
                    continue
                obj["bbox"] = bbox.tolist()
            new_annotations.append(new_annotation)
        self.annotations = new_annotations
        with open(path_to_save, 'w') as f:
            json.dump(new_annotations, f)

    def __getitem__(self, idx):
        image, annotation = super(SHOP_VRB_mask, self).__getitem__(idx)
        # print(annotation)

        img_size = self.get_img_size(idx)
        size_tup = (img_size['width'], img_size['height'])

        objects = annotation['objects']
        bboxes = []
        masks = []
        classes = []

        for obj in objects:
            rle = obj['mask']
            # mask = mask_utils.decode(rle)
            bbox = obj['bbox']
            bboxes.append(bbox)
            masks.append(rle)
            classes.append(self.name_to_id[obj['name']])

        bboxes = torch.as_tensor(bboxes).reshape(-1, 4)
        target = BoxList(bboxes, size_tup, mode='xyxy')

        masks = SegmentationMask(masks, size_tup, mode='mask')
        target.add_field('masks', masks)

        classes = torch.tensor(classes)
        target.add_field('labels', classes)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, idx

    def visualise_instance(self, idx):
        import numpy as np
        image, target, _ = self.__getitem__(idx)
        draw = ImageDraw.Draw(image)
        boxes = target.bbox
        masks = target.extra_fields['masks'].masks
        for v, m in enumerate(masks):
            mask = m.mask
            pil_mask = Image.fromarray(np.uint8(255 * mask.numpy()))
            image.paste(self.color_map[target.extra_fields['labels'][v]], (0, 0), pil_mask)
        for i in range(target.__len__()):
            draw.rectangle([boxes[i][0], boxes[i][1], boxes[i][2],
                            boxes[i][3]], fill=None, outline='red')
        image.show()

    def get_img_info(self, index):
        img_data = self.get_img_size(index)
        return img_data


def mask_to_bbox(mask, img_size):
    """Function from ns-vqa"""
    """Compute the tight bounding box of a binary mask."""
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]

    x0, y0, x1, y1 = clip_xyxy_to_image(x0, y0, x1, y1, img_size[0], img_size[1])

    return np.array((x0, y0, x1, y1), dtype=np.float32)


def clip_xyxy_to_image(x0, y0, x1, y1, height, width):
    """Clip coordinates to an image with the given height and width."""
    x0 = np.minimum(width - 1., np.maximum(0., x0))
    y0 = np.minimum(height - 1., np.maximum(0., y0))
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    return x0, y0, x1, y1
