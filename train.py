import sys
import os

from jrgan.models.mfactory import MFactory


def main(argv, argc=None):

    dataset_path = './datasets/dataset_00'
    if not os.path.isdir(dataset_path):
        raise Exception('DATASET DIT not exist!!!')

    model = MFactory.getModel(
        0,
        (128, 128, 3),
        upscale=4,
        learning_rate=1e-4,
        path_dataset=dataset_path,
        filters_gen=64,
        filters_dis=64
        )
    model.train(
        epochs=3000,
        batch_size=1,
        sample_interval=50
        )


if __name__ == '__main__':
    print('Starting app...')
    main(sys.argv, len(sys.argv))
    print('Success!')
