import definitions as defs
from iof_train.train import train
import os
# from iof_train.augment import apply_augmentations, transform1

# def augment_dataset(source_dir, output_dir, transform_func=transform1, reps_per_im=3):
#     source_dir = os.path.join(defs.DATA_DIR, source_dir)
#     output_dir = os.path.join(defs.DATA_DIR, output_dir)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     apply_augmentations(source_dir=source_dir, output_dir=output_dir,
#                         transform_func=transform_func, reps_per_im=reps_per_im,
#                         visualize=False)


def train_network(data_dir, model_id, epochs=10, batch_size=16):
    train(data_dir, model_id, epochs, batch_size)
