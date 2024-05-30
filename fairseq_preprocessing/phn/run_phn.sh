#!/bin/bash
source /home1/Amartya/envs/av/bin/activate
set -e

#  provide the input train, test, dev folder(wav.scp, text) in kaldi format 
#  and the output folder where the data will be saved in fairseq format
train=1
dev=1
test=1
# "te mt mr mg kn hi ch bn bh"
tags="te mt mr mg kn hi ch"
subset="s1"
. parse_options.sh

for tag in ${tags}; do

train_folder=/home1/Amartya/fairseq-exp/data/raw/RESPIN/train/train_${tag}_${subset}
phn_lex=/home1/Amartya/fairseq-exp/data/lexicon/${tag}_lexicon.txt
dev_folder=/home1/Amartya/fairseq-exp/data/raw/RESPIN/dev/dev_${tag}_nt
test_folder=/home1/Amartya/fairseq-exp/data/raw/RESPIN/test/test_${tag}_nt
output_folder=/home1/Amartya/fairseq-exp/data/processed/RESPIN/phn/${tag}_${subset}
pyscript=/home1/Amartya/fairseq-exp/fairseq_preprocessing/data_prep_phn.py

echo $phn_lex

mkdir -p $output_folder

if [ $train == 1 ]; then
python3 $pyscript \
    --folder  ${train_folder} \
    --phn_lex ${phn_lex} \
    --save_folder ${output_folder}  \
    --tag train \
    --wav_prep \
    --text_prep \
    --lexicon \
    --save_dict 
fi

if [ $dev == 1 ]; then
python3 $pyscript \
    --folder ${dev_folder} \
    --phn_lex ${phn_lex} \
    --save_folder ${output_folder} \
    --tag dev \
    --wav_prep \
    --text_prep  \
    # --dialect_prep
fi

if [ $test == 1 ]; then
python3 $pyscript \
    --folder ${test_folder} \
    --phn_lex ${phn_lex} \
    --save_folder ${output_folder} \
    --tag test \
    --wav_prep \
    --text_prep \
    --dialect_prep
fi
done