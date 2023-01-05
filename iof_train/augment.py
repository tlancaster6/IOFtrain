import albumentations as A
import cv2
import xmltodict
import pascal_voc_writer
from glob import glob
import os
from numpy import round
from tqdm import tqdm
from iof_train import utils

# transform1 = A.Compose(
#     [A.ShiftScaleRotate(p=1)],
#     bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=['class_labels']),
# )
#

transform1 = A.Compose(
    [A.Flip(p=0.5),
     A.ShiftScaleRotate(p=0.5),
     A.RandomRotate90(p=0.5),
     A.SafeRotate(p=0.5),
     A.BBoxSafeRandomCrop(p=0.5),
     A.GridDistortion(p=0.5),
     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
     A.Sharpen(p=0.25),
     A.OneOf([
         A.GlassBlur(p=0.25, max_delta=1),
         A.AdvancedBlur(p=0.25),
         A.MedianBlur(p=0.25),
         A.GaussianBlur(p=0.25)
     ], p=0.5)
     ],
    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=['class_labels']),
)

def read_voc_xml(file_path):
    with open(file_path, 'r') as f:
        xml_content = f.read()
        xml_dict = xmltodict.parse(xml_content)['annotation']
        filename = xml_dict['filename']
        objects = xml_dict['object']
        bboxes = [[int(o['bndbox'][key]) for key in ['xmin', 'ymin', 'xmax', 'ymax']] for o in objects]
        class_labels = [o['name'] for o in objects]
        return bboxes, class_labels, filename


def write_voc_xml(img_path, width, height, bboxes, class_labels):
    xml_path = img_path.replace('.jpg', '.xml')
    writer = pascal_voc_writer.Writer(img_path, height=height, width=width)
    for bb, cl in list(zip(bboxes, class_labels)):
        writer.addObject(cl, *bb)
    writer.save(xml_path)


def get_xml_paths(source_dir):
    xml_paths = glob(os.path.join(source_dir, '*.xml'))
    return xml_paths

def apply_augmentation(source_xml_path, transform_func, visualize=False):
    bboxes, class_labels, fname = read_voc_xml(source_xml_path)
    img_path = os.path.join(os.path.dirname(source_xml_path), fname)
    source_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    transformed = transform_func(image=source_img, bboxes=bboxes, class_labels=class_labels)
    int_bboxes = []
    for bbox in transformed['bboxes']:
        int_bboxes.append(tuple([int(round(x)) for x in bbox]))
    transformed['bboxes'] = int_bboxes
    if visualize:
        utils.visualize(transformed['image'], transformed['bboxes'], transformed['class_labels'])
    return transformed, fname


def apply_augmentations(source_dir, output_dir, transform_func, reps_per_im, visualize=False):
    xml_paths = get_xml_paths(source_dir)
    print('applying augmentations')
    for xml_p in tqdm(xml_paths):
        for rep in range(reps_per_im):
            transformed, fname,  = apply_augmentation(xml_p, transform_func, visualize=visualize)
            img_out_path = os.path.join(output_dir, f'{os.path.splitext(fname)[0]}_aug{rep}.jpg')
            cv2.imwrite(img_out_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            h, w, _ = transformed['image'].shape
            write_voc_xml(img_path=img_out_path, width=w, height=h,
                          bboxes=transformed['bboxes'],
                          class_labels=transformed['class_labels'])








