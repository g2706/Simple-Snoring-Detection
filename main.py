import argparse
from inference import run_chunked_detection

parser = argparse.ArgumentParser()
parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
parser.add_argument('--model', type=str, required=True, help='Path to model weights')
parser.add_argument('--chunk_sec', type=float, default=0.5, help='Chunk length in seconds')
parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')

args = parser.parse_args()

run_chunked_detection(args.audio, args.model, args.chunk_sec, args.threshold)
