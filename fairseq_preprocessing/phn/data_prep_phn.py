import os
import soundfile
import re
import pandas as pd
import argparse
from tqdm import tqdm
from multiprocessing import Pool

#https://github.com/facebookresearch/fairseq/issues/2819
#https://github.com/facebookresearch/fairseq/issues/2654

parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True)
parser.add_argument('--phn_lex', required=True)
parser.add_argument('--save_folder', required=True)
parser.add_argument('--tag', required=True)
parser.add_argument('--sr', default=16000)
parser.add_argument('--save_dict', action='store_true', default=False)
parser.add_argument('--nj', default=8)
parser.add_argument('--text_prep', action='store_true', default=False)
parser.add_argument('--wav_prep', action='store_true', default=False)
parser.add_argument('--dialect_prep',action='store_true', default=False)
parser.add_argument('--lexicon', action='store_true', default=False)
parser.add_argument('--dir_path', default='.')


def check_files():
    files = os.listdir(args.folder)
    assert 'wav.scp' in files
    assert 'text' in files
    assert args.tag in ['train', 'dev', 'test']
    
def create_save_path():
    os.popen(f'mkdir -p {args.save_folder}')

def extract_phn_and_word():
    phn_lex_df = pd.read_csv(args.phn_lex, sep='\t', header=None)
    phn_lex_dict = dict(zip(phn_lex_df[0], phn_lex_df[1]))
    
    text_df = pd.read_csv(os.path.join(args.folder, 'text'), sep='\t', names=['utt_id', 'text'])
    text_df.sort_values(by=['utt_id'], inplace=True)
    text_df.reset_index(drop=True, inplace=True)

    words, phns, lexicon, phn_dict = [], [], {}, {"|":1}
    for line in tqdm(text_df['text']):
        word, phn = [], []
        for item in line.split():
            if item not in phn_lex_dict:
                assert False, f'phn_lex_dict does not have {item}'
            word.append(item)  
            phn.append(phn_lex_dict[item]+' |')
            lexicon[item] = phn_lex_dict[item]
            for item in phn_lex_dict[item].split():
                if item not in phn_dict:
                    phn_dict[item]  = 1
                else:
                    phn_dict[item] += 1

        words.append(' '.join(word))
        phns.append(' '.join(phn))
        
    return words, phns, lexicon, phn_dict

def save_text_metatdata(data):
    words, phn, lexicon, phn_dict = data
    word_save_path = os.path.join(args.save_folder, args.tag+'.wrd')
    phn_save_path = os.path.join(args.save_folder, args.tag+'.phn')
    print(f'saving word metadata: {word_save_path}')    
    with open(word_save_path, 'w') as f:
        for line in words:
            f.write(line+'\n')

    print(f'saving phn metadata: {phn_save_path}')
    with open(phn_save_path, 'w') as f:
        for line in phn:
            f.write(line+'\n')
    
    if args.save_dict:
        phn_dict_save_path = os.path.join(args.save_folder, 'dict.phn.txt')
        print(f'saving phn dict: {phn_dict_save_path}')
        with open(phn_dict_save_path, 'w') as f:
            for key in phn_dict:
                if key == ' ': continue
                f.write(f'{key} {phn_dict[key]}\n')

    if args.lexicon:
        lexicon_save_path = os.path.join(args.save_folder, 'lexicon.lst')
        print(f'saving lexicon: {lexicon_save_path}')
        with open(lexicon_save_path, 'w') as f:
            for key in lexicon:
                f.write(f'{key}\t{lexicon[key]}\n')
     
def save_wav_metatdata(data):
    wavs, frames = data
    wav_save_path = os.path.join(args.save_folder, args.tag+'.tsv')
    print(f'saving wav metadata: {wav_save_path}')
    with open(wav_save_path, 'w') as f:
        f.write(args.dir_path+'\n')
        for idx in range(len(wavs)):
            f.write(f'{wavs[idx]}\t{frames[idx]}\n')

def make_manifest():
    wavs_df = pd.read_csv(os.path.join(args.folder, 'wav.scp'), sep='\t', names=['utt_id', 'path'])
    wavs_df.sort_values(by=['utt_id'], inplace=True)
    wavs_df.reset_index(drop=True, inplace=True)
    
    print(f'num of lines found:', len(wavs_df['path']))
    with Pool(args.nj) as p:
        frames = list(tqdm(p.imap(get_frames, wavs_df['path']), total=len(wavs_df['path'])))
    return wavs_df['path'], frames

def get_frames(path):
   frames = soundfile.info(path).frames
   return frames 

def make_dialect():
    with open(os.path.join(args.folder, 'utt2dial'), 'r') as f:
        lines = sorted(f.read().splitlines())
    utt2lang = ['{}\t{}'.format(re.split('[ \t]',l)[0].split('_',2)[-1], re.split('[ \t]', l )[1]) for l in lines]
    return utt2lang
    
def save_dialect(data):
    dialect_save_path = os.path.join(args.save_folder, "utt2dialect_{}".format(args.tag))
    print(f'saving dialect metadata: {dialect_save_path}')
    with open(dialect_save_path, 'w') as f:
        for i in data:
            f.write(f'{i}\n')
         
def main():
    check_files()
    create_save_path()

    if args.text_prep:
        data = extract_phn_and_word()
        save_text_metatdata(data)
    if args.wav_prep:
        data = make_manifest()
        save_wav_metatdata(data)
    if args.dialect_prep:
        data = make_dialect()
        save_dialect(data)
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    main()