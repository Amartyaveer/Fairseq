
import sys
sys.path.append('/home1/fairseq/')
from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
import torchaudio
import yaml
import argparse
import fairseq
import torch
import re
import soundfile as sf
import os
from tqdm import tqdm

args, model, generator, task = None, None, None, None

def read_file(file):
    with open(file, 'r') as f:
        lines = f.read().splitlines()[1:]
    
    for i in range(len(lines)):
        lines[i] = re.split('[:\t]', lines[i])
        
    return lines

def load_config(conf):
    global args
    # reading config
    with open(conf) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**config)

def load_model(mdl_pth):
    global model, generator, task
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([mdl_pth])
    model = model[0].to('cuda')
    generator = W2lViterbiDecoder(args, task.target_dictionary)

# use viterbi decoder
def decode(inp):
    model.eval()
    try:
        encoder_out = model(**inp, features_only=True)
        emm = model.get_normalized_probs(encoder_out, log_probs=True)
        emm = emm.detach().cpu().float()
        emm = emm.transpose(0, 1)
        out = generator.decode(emm)
        text=""
        for i in out[0][0]['tokens']:
            text+=task.target_dictionary[i]
        return text
    except:
        return "**Error**"

def get_data(file):
    if len(file) == 4:
        wav, sr = sf.read(file[0], start=int(file[1]), stop=int(file[1]) + int(file[2]))
    elif len(file) == 2:
        wav, sr = sf.read(file[0])
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    inp = {'source': torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to('cuda') , 'padding_mask':torch.zeros(wav.shape[-1]).to('cuda')}
    return inp

def write_file(file, text):
    with open(file, 'a') as f:
        f.write("%s\n" % text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_pth', type=str, required=True, help='path to the audio file')
    parser.add_argument('--conf', type=str, default='config.yaml' ,help='path to the config file')
    parser.add_argument('--mdl_pth', type=str,required=True, help='path to the model')
    parser.add_argument('--out_pth', type=str, default='hypos.txt', help='path to the output file')
    args = parser.parse_args()
    load_config(args.conf)
    load_model(args.mdl_pth)
    lines = read_file(args.tsv_pth)
    for line in tqdm(lines):
        inp = get_data(line)
        out = decode(inp)
        write_file(args.out_pth, out)
    

if __name__ == "__main__":
    main()

    






