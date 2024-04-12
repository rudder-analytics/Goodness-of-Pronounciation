# Goodness-of-Pronounciation

We use Kaldi's GOP_speechocean recipe to calculate the gop of give audio and utterence.


Download Librispeech model: https://kaldi-asr.org/models/m13

# main.py

1. Create data files for kaldi including wav.scp, utt2spk, utt2text
2. Call run_gop.sh script.
   1. Load acoustic model, language model and ivector extractor model
   2. Feature extraction: Extract MFCC, CMVN, IVECTOR
   3. Compute Log-likelihoods.
   4. Compute alignment.
   5. Using the log-likelihoods and alignment, use compute-gop to calculate gop scores.
3. calculate phone scores. As the gop scores are log likelihoods, we can convert it to regular decimal scale by calculating 10 ^ (log score).
4. Calculate overall score. Calculate overall score by taking average of all phone scores.
5. Get phone timings from exp/ali-phone.1 file.
6. Display the result in json form.

# How to run the script:

python main.py audio_file_path text

eg. python main.py MACHINE.wav MACHINE

# Output
{  
'phonemes': 'M AH SH IY N', 'avg_phone_prob_score': 0.18792639890064897, 'avg_phone_log_prob_score': -1.16205264  
}  
{  
'Word Phonemes': 'M AH SH IY N', 'Average Score': 19, 'Average Log Score': '-1.1621',  
'Phoneme Probability': 'M(0.48) AH(0.31) SH(0.00) IY(0.09) N(0.05)',  
'Phoneme Probability Extended': {'M': 48, 'AH': 31, 'SH': 0, 'IY': 9, 'N': 5},  
'Word Phone Time': "[['M', 0.02, 0.05], ['AH', 0.05, 0.06], ['SH', 0.06, 0.1], ['IY', 0.1, 0.16], ['N', 0.16, 0.2]]",  
'Word': 'MACHINE'  
}

