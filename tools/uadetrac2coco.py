import itertools
import json
import os
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree

import click
import tqdm


def iter_image_boxes_pairs(root_path, subset):
    """It reads image (frame) file names as well as their corresponding bounding
    boxes in a tensor format for faster access later on.

    Args:
        root_path (str): UA-DETRAC dataset root path.
        subset (str): Whether to read 'train' or 'test' data subset.
    """
    images_dir, annos_dir = _deduce_images_and_annos_paths(root_path, subset)

    for seq_num, seq_dir in enumerate(images_dir.iterdir(), start=1):
        if seq_num > 1:
            break

        xml_file_name = seq_dir.stem + '_v3.xml'
        xml_file_path = str(annos_dir / xml_file_name)

        image_boxes_map = dict(_iter_seq_boxes(xml_file_path))

        for image_num, image_file_path in _iter_seq_image_file_paths(seq_dir):
            if image_num > 100:
                break

            boxes = image_boxes_map.get(image_num)
            if boxes is not None:
                yield seq_num, image_num, image_file_path, boxes


def _iter_seq_image_file_paths(seq_dir):
    """Iterates over image file names for a specific sequence from the
    UA-DETRAC dataset.

    Args:
        seq_dir (pathlib.Path): Sequence directory path.

    Yields:
        Tuple[int, str]: Tuple containing image (frame) number and
        the corresponding file path.
    """
    image_num_path_pairs = [
        (int(p.stem[-5:]), str(p)) for p in seq_dir.iterdir()
    ]
    yield from iter(sorted(image_num_path_pairs))


def _iter_seq_boxes(xml_file_path):
    """Iterates over a sequence of bounding boxes contained within a
    specific XML file corresponding to some sequence from the UA-DETRAC
    dataset.

    Args:
        xml_file_path (str): Sequence specification XML file path.

    Yields:
        Tuple[int, List[Tuple[float, float, float, float]]]: A tuple
        containing the frame number and the list of bounding boxes in a
        xywh format.
    """
    tree = ElementTree.parse(xml_file_path)
    root = tree.getroot()

    for frame in root.findall('./frame'):
        frame_num = int(frame.attrib['num'])
        boxes = []

        for target in frame.findall('.//target'):
            box_attr = target.find('box').attrib
            
            x = float(box_attr['left'])
            y = float(box_attr['top'])
            w = float(box_attr['width'])
            h = float(box_attr['height'])

            box = [x, y, w, h]
            boxes.append(box)
        
        yield frame_num, boxes


def _deduce_images_and_annos_paths(root_path, subset):
    """Deduces paths for images and annotations. It returns the root path
    that contains all the sequences belonging to the specific subset.

    Args:
        root_path (str): Root directory path to the UA-DETRAC dataset.
        subset (str): Data subset type ('train' or 'test').

    Returns:
        Tuple[pathlib.Path, pathlib.Path]: Directory paths for images and
            annotations.
    """
    assert subset in ('train', 'test')

    subset = subset.capitalize()
    root_dir = Path(root_path)

    images_idr = root_dir / ('Insight-MVT_Annotation_' + subset)
    annos_dir = root_dir / 'DETRAC_public' / ('540p-' + subset)

    return images_idr, annos_dir


def export_dataset(dataset_input_dir, dataset_output_dir, dataset_name, subset):
    dataset_output_dir = os.path.join(dataset_output_dir, dataset_name)
    images_dir_path = os.path.join(dataset_output_dir, 'data')

    if os.path.exists(dataset_output_dir):
        shutil.rmtree(dataset_output_dir)
    os.makedirs(images_dir_path)

    data = {}
    data['info'] = {
        'year': '', 'version': '', 'description': 'UA-DETRAC dataset',
        'contributor': '', 'url': '', 'date_created': ''
    }
    data['licenses'] = []
    vehicle_cat_id = 0
    data['categories'] = [
        {'id': vehicle_cat_id, 'name': 'vehicle', 'supercategory': 'vehicle'}
    ]
    images, annotations = [], []
    data['images'] = images
    data['annotations'] = annotations

    image_id_gen = itertools.count()
    annotation_id_gen = itertools.count()

    image_height, image_width = 540, 960

    tqdm_pbar = tqdm.tqdm(file=sys.stdout)
    with tqdm_pbar as pbar:
        data_iter = iter_image_boxes_pairs(dataset_input_dir, subset)
        for seq_num, image_num, image_file_path, boxes in data_iter:
            pbar.set_description(
                f"processing seq. {seq_num}, sample {image_file_path}"
            )

            dst_file_name = f'{seq_num:02d}_{image_num:04d}.jpg'
            dst_file_path = os.path.join(images_dir_path, dst_file_name)
            shutil.copyfile(image_file_path, dst_file_path)

            image_id = next(image_id_gen)
            image_data = {
                'id': image_id,
                'file_name': dst_file_name,
                'seq_num': seq_num,
                'image_num': image_num,
                'height': image_height,
                'width': image_width,
            }
            images.append(image_data)

            for box in boxes:
                annotation = {
                    'id': next(annotation_id_gen),
                    'image_id': image_id,
                    'category_id': vehicle_cat_id,
                    'bbox': box,
                    'area': box[2] * box[3],
                    'iscrowd': 0,
                }
                annotations.append(annotation)

            pbar.update()
    
    json_file_path = os.path.join(dataset_output_dir, 'annotations.json')
    with open(json_file_path, 'wt') as out_file:
        json.dump(data, out_file)


@click.command()
@click.argument('dataset_root_dir')
@click.argument('export_dir')
@click.option(
    '--dataset-name', default='UA-DETRAC_COCO', show_default=True,
    help="The new dataset name."
)
@click.option(
    '-s', '--subset', default='test', show_default=True,
    help="Data subset type."
)
def main(dataset_root_dir, export_dir, dataset_name, subset):
    export_dataset(dataset_root_dir, export_dir, dataset_name, subset)
    return 0


if __name__ == '__main__':
    sys.exit(main())
