import torch
import torch.utils.data as data
import os
import json
from PIL import Image, ImageDraw
import pycocotools.mask as mask_utils
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import numpy as np

'''
Create COCO Detection style dataset from CLEVR-Ref+ generated images
(with bounding boxes and segmentation masks).
Compatible with MaskRCNN Benchmark
'''


class CLEVR_mini(data.Dataset):
    # Baseline for different tasks
    def __init__(self, path, transforms=None):
        images_dir = os.path.join(path, 'images')
        annotation_file = os.path.join(path, 'CLEVR_mini_coco_anns.json')
        assert os.path.isfile(annotation_file), 'Annotation file not found'

        with open(annotation_file) as f:
            annotations = json.load(f)['scenes']

        self.ids = [obj['image_index'] for obj in annotations]
        file_names = [os.path.join(images_dir, obj['image_filename']) for obj in annotations]
        self.ids_to_names = dict(zip(self.ids, file_names))

        self.shape_to_id = {'cube': 1, 'cylinder': 2, 'sphere': 3}
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


class CLEVR_mini_segmentation(CLEVR_mini):
    # Initialise dataset from either original file or preprocessed for detection
    def __init__(self, path, transforms=None, comb_class=True):
        preprocessed_file = os.path.join(
            path, 'CLEVR_mini_coco_anns') + '_detection_preprocessed.json'
        if not os.path.isfile(preprocessed_file):
            # if you don't have file use parent's annotations
            super(CLEVR_mini_segmentation, self).__init__(path, comb_class)
            print(preprocessed_file)
            self.preprocess_dataset(preprocessed_file)
        with open(preprocessed_file) as f:
            annotations = json.load(f)
        images_dir = os.path.join(path, 'images')
        self.ids = [obj['image_index'] for obj in annotations]
        file_names = [os.path.join(images_dir, obj['image_filename']) for obj in annotations]
        if comb_class:
            colors = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
            materials = ['rubber', 'metal']
            shapes = ['cube', 'cylinder', 'sphere']
            idx = 1
            prop_to_id = {}
            for c in colors:
                for m in materials:
                    for s in shapes:
                        prop_to_id[(c, m, s)] = idx
                        idx += 1
        else:
            prop_to_id = {'cube': 1, 'cylinder': 2, 'sphere': 3}
        self.prop_to_id = prop_to_id
        self.comb_class = comb_class
        self.ids_to_names = dict(zip(self.ids, file_names))
        self.annotations = annotations
        self.transforms = transforms
        # Just a list of some RGB colors, add more if you have more classes
        self.color_map = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                          for i in range(len(self.prop_to_id) + 1)]

    # Preprocess dataset only for idxs, filenames, labels, boxes, masks
    def preprocess_dataset(self, path_to_save):
        print("Preprocessed annotation file not found")
        print("Preprocessing dataset for COCO detection format annotations...")
        from tqdm import tqdm
        new_annotations = []
        for idx in tqdm(range(self.__len__())):
            annotation = self.annotations[idx]
            new_annotation = {'image_index': annotation['image_index'],
                              'image_filename': annotation['image_filename'], 'objects': []}
            new_annotation['directions'] = annotation['directions']
            img_size = annotation['objects'][0]['mask']['size']
            for obj in annotation['objects']:
                rle_mask = obj['mask']
                mask = mask_utils.decode(rle_mask)
                bbox = mask_to_bbox(mask, img_size)
                area = mask.sum()
                if not (area > 0 and bbox[2] > bbox[0] and bbox[3] > bbox[1]):
                    continue
                new_annotation['objects'].append(
                    {'shape': obj['shape'], 'bbox': bbox.tolist(), 'mask': rle_mask, 'color': obj['color'], 'size': obj['size'], 'material': obj['material'], '3d_coords': obj['3d_coords']})
            new_annotations.append(new_annotation)
        self.annotations = new_annotations
        with open(path_to_save, 'w') as f:
            json.dump(new_annotations, f)

    def __getitem__(self, idx):
        image, annotation = super(CLEVR_mini_segmentation, self).__getitem__(idx)
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
            # masks.append(mask)
            masks.append(rle)
            if self.comb_class:
                classes.append(self.prop_to_id[(obj['color'], obj['material'], obj['shape'])])
            else:
                classes.append(self.prop_to_id[obj['shape']])

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


class CLEVR_mini_segmentation_inference(CLEVR_mini_segmentation):
    # Initialise dataset from either original file or preprocessed for detection
    def __init__(self, path, transforms):
        file = os.path.join(
            path, 'CLEVR_mini_coco_anns') + '_detection_inference.json'

        assert os.path.isfile(file)

        with open(file) as f:
            annotations = json.load(f)
        images_dir = os.path.join(path, 'images')
        self.ids = [obj['image_index'] for obj in annotations]
        file_names = [os.path.join(images_dir, obj['image_filename']) for obj in annotations]
        self.ids_to_names = dict(zip(self.ids, file_names))
        self.annotations = annotations
        self.transforms = transforms
        # Just a list of some RGB colors, add more if you have more classes
        self.color_map = [(0, 0, 205), (227, 207, 87), (127, 255, 0),
                          (118, 238, 198), (61, 145, 64), (255, 127, 0)]

    def __getitem__(self, idx):
        image, annotation = super(CLEVR_mini_segmentation, self).__getitem__(idx)
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
            classes.append(obj['label'])

        bboxes = torch.as_tensor(bboxes).reshape(-1, 4)
        target = BoxList(bboxes, size_tup, mode='xyxy')

        masks = SegmentationMask(masks, size_tup)
        target.add_field('masks', masks)

        classes = torch.tensor(classes)
        target.add_field('labels', classes)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, idx


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
