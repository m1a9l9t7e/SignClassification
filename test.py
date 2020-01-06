import logging
from settings import Settings
import argparse
import warnings
import util
import model_util

logging.getLogger('tensorflow').disabled = True
logging.getLogger('numpy').disabled = True
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# Path to folder containing images/videos. This need to be provided
parser.add_argument('--data', dest='path_to_data', required=True)
# Path to settings, if left blank, the latest model will be downloaded automatically!
parser.add_argument('--settings', dest='path_to_settings', default=None)
args = parser.parse_args()

if args.path_to_settings is None:
    settings = util.import_latest_model()
else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

# read images and transform according to settings
images = util.read_any_data(args.path_to_data, settings=settings)

# get model path and restore and evaluate model
path_to_model = settings.get_setting_by_name('model_save_path')
model_util.evaluate_images(settings, path_to_model, images, show=True)
