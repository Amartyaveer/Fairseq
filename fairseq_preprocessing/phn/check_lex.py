import argparse
from tqdm import tqdm
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--tag', required=True)
parser.add_argument('--folder', required=True)
parser.add_argument('--phn_lex', required=True)
args = parser.parse_args()

phn_lex_df = pd.read_csv(args.phn_lex, sep='\t', header=None)
phn_lex_dict = dict(zip(phn_lex_df[0], phn_lex_df[1]))

text_df = pd.read_csv(os.path.join(args.folder, 'text'), sep='\t', names=['utt_id', 'text'])
text_df.sort_values(by=['utt_id'], inplace=True)


missing_phn = []
for line in tqdm(text_df['text']):
    for item in line.split():
        if item not in phn_lex_dict:
            missing_phn.append(item)


with open(f'{args.tag}_missing_phn.txt', 'a') as f:
    for item in set(missing_phn):
        f.write(f'{item}\n')