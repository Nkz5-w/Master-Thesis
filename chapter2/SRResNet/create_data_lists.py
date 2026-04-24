import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['datasets/train_ckm_pathloss'],
                      test_folders=['datasets/test_ckm_pathloss'],
                      min_size=100,
                      output_folder='./')
