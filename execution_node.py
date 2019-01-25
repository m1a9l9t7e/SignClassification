import argparse
import warnings
import util
import tf_util
from data import OrderedBatchProvider


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest='path_to_data', help='Path to folder containing images/videos. This need to be provided')
parser.add_argument('--path_to_import', dest='path_to_import', default='.', help='determines where model is imported to')

args = parser.parse_args()

settings = util.import_latest_model()
print('regular without labels')
path_to_data = args.path_to_data
data = util.read_any_data(path_to_data, settings=settings)
test_data_provider = OrderedBatchProvider(data, [], settings.get_setting_by_name('batch_size'), [])
tf_util.execute_frozen_model(settings, test_data_provider)
