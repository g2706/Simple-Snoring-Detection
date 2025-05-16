import argparse
from inference import run_chunked_detection

parser = argparse.ArgumentParser()
parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
parser.add_argument('--model', type=str, required=True, help='Path to model weights')
parser.add_argument('--chunk_sec', type=float, default=0.5, help='Chunk length in seconds')
parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')

args = parser.parse_args()

run_chunked_detection(args.audio, args.model, args.chunk_sec, args.threshold)

dic_sty = ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid',
           'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot',
           'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind',
           'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid',
           'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper',
           'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks',
           'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']