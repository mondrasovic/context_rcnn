import sys

import fiftyone as fo

def _init_data_indices(self, root_path, subset):
    """Initializes data indices to faster access. It reads image (frame)
    file names as well as their corresponding bounding boxes in a tensor
    format for faster access later on.

    Args:
        root_path (str): UA-DETRAC dataset root path.
        subset (str): Whether to read 'train' or 'test' data subset.
    """
    images_dir, annos_dir = self._deduce_images_and_annos_paths(
        root_path, subset
    )

    for seq_idx, seq_dir in enumerate(images_dir.iterdir()):
        xml_file_name = seq_dir.stem + '_v3.xml'
        xml_file_path = str(annos_dir / xml_file_name)

        image_files_iter = self._iter_seq_image_file_paths(seq_dir)
        image_boxes_iter = self._iter_seq_boxes(xml_file_path)
        data_iter = itertools.zip_longest(
            image_files_iter, image_boxes_iter
        )

        seq_image_file_paths = []
        seq_image_boxes = []

        for image_idx, (files_iter_data, boxes_iter_data) in enumerate(
            data_iter
        ):
            assert files_iter_data is not None
            assert boxes_iter_data is not None

            image_num_1, image_file_path = files_iter_data
            image_num_2, boxes = boxes_iter_data
            assert image_num_1 == image_num_2 

            seq_image_file_paths.append(image_file_path)
            seq_image_boxes.append(boxes)

            seq_boxes_idx = self._SeqBoxesIndex(seq_idx, image_idx)
            self._global_to_local_seq_image_idxs.append(seq_boxes_idx)
        
        self._seq_image_paths.append(seq_image_file_paths)
        self._seq_boxes.append(seq_image_boxes)
def main():
    dataset = fo.Dataset(name='UA-DETRAC-COCO')
    dataset.persistent = True

    return 0


if __name__ == '__main__':
    sys.exit(main())
