#!/usr/bin/env bash

# Copyright 2019 Junbo Zhang
# Apache 2.0

# This script shows how to calculate Goodness of Pronunciation (GOP) and
# extract phone-level pronunciation feature for mispronunciations detection
# tasks. Read ../README.md or the following paper for details:
#
# "Hu et al., Improved mispronunciation detection with deep neural network
# trained acoustic models and transfer learning based logistic regression
# classifiers, 2015."

# You might not want to do this for interactive shells.


set -e
. ./cmd.sh
. ./path.sh
. parse_options.sh

# This script assumes the following paths exist which are from LIBRISPEECH recipe and pre-trained model.
#model=exp/chain_cleaned/tdnn1f_2048_sp_bi
#extractor=exp/nnet3_cleaned/extractor
#lang=data/lang_std_big_v5
conf=conf/

#model=/home/rau047/tuda/de_900k_nnet3chain_tdnn1f_2048_sp_bi/
#extractor=/home/rau047/tuda/de_900k_nnet3chain_tdnn1f_2048_sp_bi/ivector_extractor/
#lang=/home/rau047/tuda/data/lang_std_big_v6_const_arpa/
#conf=/home/rau047/tuda/de_900k_nnet3chain_tdnn1f_2048_sp_bi/conf/

extractor=/home/rau047/libreespeech2/0013_librispeech_v1_extractor/exp/nnet3_cleaned/extractor
model=/home/rau047/libreespeech2/0013_librispeech_v1_chain/exp/chain_cleaned/tdnn_1d_sp
lang=/home/rau047/libreespeech2/0013_librispeech_v1_lm/data/lang_test_tgsmall
#conf=/home/rau047/libreespeech/conf/


for d in $model $ivector $lang $conf ; do
  [ ! -d $d ] && echo "$0: no such path $d" && exit 1;
done

# Global configurations
stage=0
nj=1

# Prepare testdata directory (with wav.scp, text, utt2spk and spk2utt files)
# testdata=test_10short 
#testdata=test_native_vs_others
testdata=$1
dir=exp/gop_$testdata


# Feature extraction like MFCC,CMVN,IVECTOR for a given test data
if [ $stage -le 0 ]; then
  steps/make_mfcc.sh --nj $nj --mfcc-config $conf/mfcc_hires.conf \
     data/$testdata exp/make_hires/$testdata $mfccdir

  steps/compute_cmvn_stats.sh data/$testdata exp/make_hires/$testdata $mfccdir
     utils/fix_data_dir.sh data/$testdata

  steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
      data/$testdata $extractor exp/nnet3_cleaned/ivectors_${testdata}

fi

# Compute Log-likelihoods
if [ $stage -le 1 ]; then
  steps/nnet3/compute_output.sh --cmd "$cmd" --nj $nj \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_${testdata} data/$testdata $model exp/probs_$testdata
fi

# Compute alignment
if [ $stage -le 2 ]; then
  steps/nnet3/align.sh --cmd "$cmd" --nj $nj --use_gpu false \
    --online_ivector_dir exp/nnet3_cleaned/ivectors_${testdata} data/$testdata $lang $model $dir
fi


if [ $stage -le 3 ]; then
  # make a map which converts phones to "pure-phones"
  # "pure-phone" means the phone whose stress and pos-in-word markers are ignored
  # eg. AE1_B --> AE, EH2_S --> EH, SIL --> SIL
  local/remove_phone_markers.pl $lang/phones.txt $dir/phones-pure.txt \
    $dir/phone-to-pure-phone.int

  # Convert transition-id to pure-phone id
  $cmd JOB=1:$nj $dir/log/ali_to_phones.JOB.log \
    ali-to-phones --per-frame=true $model/final.mdl "ark,t:gunzip -c $dir/ali.JOB.gz|" \
       "ark,t:|gzip -c >$dir/ali-phone.JOB.gz" || exit 1;
  
fi

if [ $stage -le 4 ]; then
  # The outputs of the binary compute-gop are the GOPs and the phone-level features.
  #
  # An example of the GOP result (extracted from "ark,t:$dir/gop.3.txt"):
  # 4446-2273-0031 [ 1 0 ] [ 12 0 ] [ 27 -5.382001 ] [ 40 -13.91807 ] [ 1 -0.2555897 ] \
  #                [ 21 -0.2897284 ] [ 5 0 ] [ 31 0 ] [ 33 0 ] [ 3 -11.43557 ] [ 25 0 ] \
  #                [ 16 0 ] [ 30 -0.03224623 ] [ 5 0 ] [ 25 0 ] [ 33 0 ] [ 1 0 ]
  # It is in the posterior format, where each pair stands for [pure-phone-index gop-value].
  # For example, [ 27 -5.382001 ] means the GOP of the pure-phone 27 (it corresponds to the
  # phone "OW", according to "$dir/phones-pure.txt") is -5.382001, indicating the audio
  # segment of this phone should be a mispronunciation.
  #
  # The phone-level features are in matrix format:
  # 4446-2273-0031  [ -0.2462088 -10.20292 -11.35369 ...
  #                   -8.584108 -7.629755 -13.04877 ...
  #                   ...
  #                   ... ]
  # The row number is the phone number of the utterance. In this case, it is 17.
  # The column number is 2 * (pure-phone set size), as the feature is consist of LLR + LPR.
  # The phone-level features can be used to train a classifier with human labels. See Hu's
  # paper for detail.
  $cmd JOB=1:$nj $dir/log/compute_gop.JOB.log \
    compute-gop --phone-map=$dir/phone-to-pure-phone.int $model/final.mdl \
      "ark,t:gunzip -c $dir/ali-phone.JOB.gz|" \
      "ark:exp/probs_$testdata/output.JOB.ark" \
      "ark,t:$dir/gop.JOB.txt" "ark,t:$dir/phonefeat.JOB.txt"   || exit 1;
  echo "Done compute-gop, the results: \"$dir/gop.<JOB>.txt\" in posterior format."

  # We set -5 as a universal empirical threshold here. You can also determine multiple phone
  # dependent thresholds based on the human-labeled mispronunciation data.
  echo "The phones whose gop values less than -5 could be treated as mispronunciations."
fi
