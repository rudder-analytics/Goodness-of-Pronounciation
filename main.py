#import libraries
import subprocess
import os
import re
import numpy as np
import sys

CWD = '/home/rau047/kaldi/egs/gop_speechocean762/s5'

FRAME_SHIFT = 10
#Dir where file that contains phoneme start times and end times. 
PHONE_TIMES_DIR = os.path.join(CWD, 'exp', 'gop_test_api')

#function to write a file.
def write_file(file_name, text_lines):
    txt = '\n'.join(text_lines)
    txt = txt + '\n'
    with open(file_name, 'w') as f:
        f.write(txt)

#create scp file.
# scp file needs to be manully created/updated. It stores speaker id and path to spoken utternece.
#this file is required by Kaldi gop script
def create_wav_scp_file(data_dir, wav_file):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    scp_file = os.path.join(data_dir, 'wav.scp')
    text_lines = [f'1 {wav_file}']
    write_file(scp_file, text_lines)

#text file needs to be manully created/updated
#text file stores speaker id and spoken text.
#this file is required by Kaldi gop script
def create_text_file(data_dir, transcript):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    text_file = os.path.join(data_dir, 'text')
    #transcript = transcript.upper()
    text_lines = [f'1 {transcript}']
    write_file(text_file, text_lines)

#utt2spk file needs to be manully created/updated.
#utt2spk file contains utterence id to speaker id mapping.
#this file is required by Kaldi gop script
def create_utt2spk_file(data_dir):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    utt2spk_file = os.path.join(data_dir, 'utt2spk')
    text_lines = [f'1 1']
    write_file(utt2spk_file, text_lines)


#spk2utt file needs to be manully created/updated.
#spk2utt file contains utterence id to speaker id mapping.
#this file is required by Kaldi gop script    
def create_spk2utt_file(data_dir):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    utt2spk_file = os.path.join(data_dir, 'spk2utt')
    text_lines = [f'1 1']
    write_file(utt2spk_file, text_lines)

#This function creates all the required files required by Kaldi gop script.
def create_data_dir(data_dir, wav_file, transcript):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    create_wav_scp_file(data_dir, wav_file)
    create_text_file(data_dir, transcript)
    create_spk2utt_file(data_dir)
    create_utt2spk_file(data_dir)

#this function reads the file line by line.
def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# parse_phones function returns phone and phone id.
def parse_phones(filename):
    phone_lines = read_file(filename)
    phone2id = dict()
    
    id2phone = dict() 
    #print("phone_lines",phone_lines)
    #if "29" in phone_lines:
    	#print("29 removed")
    	#phone_lines.remove("29")
    #print("phone_lines:", phone_lines)
     
    for pl in phone_lines:
        if filename == '/home/rau047/kaldi/egs/gop_speechocean762/s5/exp/gop_test_api/phone-to-pure-phone.int':
            phones_ids = pl.split(' ')
            #print(phones_ids)
        else:
            phones_ids = pl.split('\t')
        #print(phones_ids)
        ph = phones_ids[0].strip()
        idx = phones_ids[1].strip()
        #print(idx)
        phone2id[ph] = idx
        id2phone[idx] = ph
        #if filename == '/home/ubuntu/kaldi/egs/gop_speechocean762/s5/exp/gop_test_api_german/ali-phone.1':
        #phone2id['BLANK'] = "29"
        #id2phone["29"] = 'BLANK'
        
        #phone2id['73'] = "Sh"
        #id2phone["S"] = '73'
        #id2phone.update({"73":"Sh"})
        #phone2id['77'] = "Ch"
        #id2phone.update({"77":"Ch"})
        #id2phone["29"] = 'BLANK'
        
    return phone2id, id2phone

#Get gop scores from gop.1.txt file. The format is speaker id [phone_id gop_score].
# [ 27 -5.382001 ] means the GOP of the pure-phone 27 (it corresponds to the phone "OW", according to "$dir/phones-pure.txt") is -5.382001

def get_scores(gop_dir):
    # gop_op_file contains file path to file that stores log likelihoods.
    gop_op_file = os.path.join(gop_dir, 'gop.1.txt')
    # phones-pure.txt stores phone to phone id mapping.
    phone_file = os.path.join(gop_dir, 'phones-pure.txt')
    # call to parse_phone function. It returns phone and phone id.
    phone2id, id2phone = parse_phones(phone_file)
    #read gop scores file.
    lines = read_file(gop_op_file)
    phone_scores = []
    for line in lines:
        matches = re.findall(r'\[.+?]', line)
        for m in matches:
            m = m[1:-1].strip()
            m_list = m.split()
            ph_id = m_list[0]
            ph_conf = m_list[1]
            ph_prob = np.power(10, float(ph_conf))
            #id2phone[ph_id] = id2phone[ph_id].replace("'","")
            #print(id2phone[ph_id])
            temp = id2phone[ph_id]
            if id2phone[ph_id]=="y:":
                temp = id2phone[ph_id].replace("y:","eu")
            if id2phone[ph_id]=="'y:":
                temp = id2phone[ph_id].replace("'y:","oe")
            temp01 = temp.replace(":","")
            temp1 = temp01.replace("@","e")
            tpl = (ph_id, temp1.replace("'",""), ph_conf, str(ph_prob))
            phone_scores.append(tpl)
    print("phone_scores", phone_scores)
    #get phone scores.
    return phone_scores

#calculate overall scores.
#returns list of phonemes, average phone probability score and avg_phone_log_prob_score.
def get_overall_score(phone_scores):
    p_ctr = 0
    u_prob = 0
    u_conf = 0
    u_phones = []
    for i, s in enumerate(phone_scores):
        if i == 0 and s[1] == 'SIL':
            continue
        if i == len(phone_scores)-1 and s[1] == 'SIL':
            continue
        p_ctr += 1
        u_prob += float(s[3])
        u_conf += float(s[2])
        u_phones.append(s[1])
        # print(s)
    if p_ctr == 0:
        u_prob_overall = 0
        u_conf_overall = 0
    else:
        u_prob_overall = u_prob/p_ctr
        u_conf_overall = u_conf/p_ctr
    utterance = dict()
    utterance['phonemes'] = ' '.join(u_phones)
    utterance['avg_phone_prob_score'] = u_prob_overall
    utterance['avg_phone_log_prob_score'] = u_conf_overall
    return utterance

#get formatted scores. Take log of log likelihoods to get original scores. 
def get_formatted_score(utterance,transcript):
    avg_prob = utterance['avg_phone_prob_score']
    avg_log_prob = utterance['avg_phone_log_prob_score']
    phonemes = utterance['phonemes']
    phone_scores = utterance['phone_scores']
    #print("in formatting: ", phone_scores)
    p_s_str = []
    p_s_dict = {}
    for i,p in enumerate(phone_scores):
        if i == 0 or i == len(phone_scores) - 1:
            continue
        p_s_str.append(f'{p[1]}({float(p[3]):.2f})')
        p_s_dict[p[1]] = round(float(p[3])*100)
        
    p_str = ' '.join(p_s_str)
    o_str = f'Word Phonemes: {phonemes}\nAverage Score: {avg_prob:.4f}\nAverage Log Score:{avg_log_prob:.4f}\n' \
            f'Phoneme Probability: {p_str} '
    o_str2 = dict()
    o_str2['Word Phonemes'] = phonemes
    o_str2['Average Score'] = round(avg_prob*100)
    o_str2['Average Log Score'] = f'{avg_log_prob:.4f}'
    o_str2['Phoneme Probability'] = p_str
    o_str2['Phoneme Probability Extended'] = p_s_dict
    o_str2['Word Phone Time'] = f"{utterance['phone_durations']}"
    o_str2['Word Phone Time Extended'] = utterance['phone_durations']    
    o_str2['Word'] = transcript
    return o_str2

#this function converts uploaded wav file into the format required by Kaldi. 
# The file is converted using sox subdirectory.
# the file must in wav format, should be a mono channel, should be converted 16-bit PCM and should have min frequency of 16 KHz.
def covert_to_wav(file_path):
    ext = os.path.splitext(file_path)[-1]
    d_file = file_path.replace(ext, '_converted.wav')
    subprocess.call(["sox", file_path, "-r 16000", "-c 1", "-b 16", d_file])
    #os.remove(file_path)
    return d_file


def get_phone_timings(dir_path):


    fp = os.path.join(dir_path, 'ali-phone.1')    
    mp = os.path.join(dir_path, 'phone-to-pure-phone.int')  
    if os.path.isfile(fp):
        os.remove(fp)
    pfp = os.path.join(dir_path, 'phones-pure.txt')
    subprocess.call(["gunzip", "ali-phone.1.gz"], cwd=dir_path)


    phone2id, id2phone = parse_phones(pfp)
    phone2purephone, purephone2phone = parse_phones(mp)
    
    #print("phone2purephone: ",phone2purephone ,', purephone2phone: ', purephone2phone) 
    #print("type of phone2id: ",type(phone2id), ', type of id2phone: ', type(id2phone))

    phone_lines = read_file(fp)
    #pure_phone_lines = read_file(mp)
    #print("pure_phone_lines: " , pure_phone_lines)
    #print("type of phone_lines: ",type(phone_lines))
    phone_durations = []
    count = 0
    for line in phone_lines:
        count+=1
        #print(count)
        line = line.strip()
        #print("line: " ,line)
        phone_tokens = line.split()
        #print("phone_tokens: " ,phone_tokens)
        phone_tokens = phone_tokens[1:]
        #print("phone_tokens: ", phone_tokens)
        for i in range(len(phone_tokens)):
            phone_tokens[i] = phone2purephone[phone_tokens[i]]
        prev_phone = ''
        #print("phone_tokens: ", phone_tokens)
        
        for i, p in enumerate(phone_tokens):
            phone_time = i * FRAME_SHIFT / 1000
            #print("phone_time: ", phone_time)
            phone = id2phone[p]
            #print("phone: ", phone)
            if phone == prev_phone:
                continue

            if phone_durations:
                phone_durations[-1][-1] = phone_time
                #print("phone_durations : ", phone_durations)
                phone_durations.append([phone, phone_time, -1])
                #print("phone_durations : ", phone_durations) 
            else:
                phone_durations.append([phone, phone_time, -1])
                #print("phone_durations : ", phone_durations)
            prev_phone = phone
            #print("prev_phone: ", prev_phone)
    print(phone_durations[1:-1])    

                
    return phone_durations[1:-1]

#this is the main function which is called in gop_demo_app_test.py file.
#this function accepts 3 args: path to wav file. transcript and path to data dir where all the kaldi related files are stored.
def run_gop(wav_file, transcript, data_dir):
    #path where all kaldi related files are stored.
    data_dir_path = os.path.abspath(data_dir)
    #path to wav file
    wav_file = os.path.abspath(wav_file)
    #convert the uploaded wav file to required format required by kaldi gop scripts.
    wav_file = covert_to_wav(wav_file)
    print(data_dir_path)
    #create all the files files required by Kaldi gop module.
    create_data_dir(data_dir_path, wav_file, transcript)
    #call to run.sh shell script which is an actual kaldi file which computes the gop scores.
    subprocess.call(["./run_gop.sh", "test_api"], cwd=CWD)
    #path to a file that stores gop scores.
    op_dir = os.path.join(CWD, 'exp', 'gop_test_api')
    
    #calculate phone scores.
    phone_scores = get_scores(op_dir)
    print(phone_scores)
    #get overall scores.
    utterance_score = get_overall_score(phone_scores)
    print(utterance_score)
    utterance_score['phone_scores'] = phone_scores
    # get phoneme durations. - start time and end time of phonemes.
    phone_durations = get_phone_timings(PHONE_TIMES_DIR)
    # print(phone_durations)
    utterance_score['phone_durations'] = phone_durations
    #print(utterance_score)
    # get formatted scores.
    f_str = get_formatted_score(utterance_score,transcript)
    #return gop result in json format and path to wav file bach to gop_demo_app_test.py file.
    print(f_str)

with open("main_log.txt", "w") as f:
	wav_file = list(sys.argv)[1]
	transcript = list(sys.argv)[2]
	data_dir = '/home/rau047/kaldi/egs/gop_speechocean762/s5/data/test_api'
	
	run_gop(wav_file, transcript, data_dir)

