# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:10:44 2021

@author: SNT
"""

import re
import numpy as np
import os, shutil, glob
import cv2
import warnings
from tqdm import tqdm

#%%############################################################################
'''                      FUNCTION detect elan files                        '''
###############################################################################

def detect_elan_files(folder_with_elan_files : str):
    # DETECTING ELAN FILES IN THE FOLDER AND SUBFOLDERS
    elan_file_list = [elan for elan in glob.glob(folder_with_elan_files+'/**/*', recursive = True) +
                        glob.glob(folder_with_elan_files+'/*', recursive = True) if '.eaf' in elan]
    elan_file_list = list(set(elan_file_list))
    elan_file_list.sort()
    number_elan_detected = len(elan_file_list)
    print('[-] {} elan files were detected in {}'.format(number_elan_detected, folder_with_elan_files))
    return elan_file_list


#%%############################################################################
'''                     FUNCTION parse time slots                           '''
###############################################################################
def parse_time_slots(elan_text):

    time_units = elan_text.split('TIME_UNITS="')[-1].split('"')[0]           
    time_order = elan_text.split('<TIME_ORDER>')[-1].split('</TIME_ORDER>')[0]
    time_slots = [ts for ts in time_order.split('\n') if 'TIME_SLOT_ID' in ts]
    
    
    
    time_slots_dict = dict(zip(                                                  
                [ts.split('TIME_SLOT_ID="')[1].split('"')[0] for ts in time_slots],
                [float(ts.split('TIME_VALUE="')[1].split('"')[0]) for ts in time_slots]               
                               ))
    time_slots_dict['time_units'] = time_units

    return time_slots_dict


#%%############################################################################
'''                     FUNCTION parse annotations                          '''
###############################################################################
def parse_annotations(elan_text):

    annotations = re.findall(r'<ANNOTATION>\s*((?:.|\n)*?)</ANNOTATION>', elan_text)
    annotations = [ann for ann in annotations if '<ANNOTATION_VALUE>' in ann]# and 'REF_ANNOTATION' not in ann]
    annotations_tuple = (
        dict(zip([ann.split('ANNOTATION_ID="')[1].split('"')[0] for ann in annotations],
                 zip([ann.split('TIME_SLOT_REF1="')[1].split('"')[0] if 'TIME_SLOT_REF1' in ann else '' for ann in annotations],
        [ann.split('TIME_SLOT_REF2="')[1].split('"')[0] if 'TIME_SLOT_REF2' in ann else '' for ann in annotations],
        [ann.split('<ANNOTATION_VALUE>')[1].split('</ANNOTATION_VALUE>')[0].replace('\n', '').strip() for ann in annotations],
        [ann.split('ANNOTATION_REF="')[1].split('"')[0] if 'ANNOTATION_REF' in ann else '' for ann in annotations])
        )
        ))
    
    return annotations_tuple


#%%############################################################################
'''                     FUNCTION parse ref annotations                      '''
###############################################################################

def parse_tier_header(tier_header):
    to_clean = ['<TIER', '>']
    for c in to_clean:
        tier_header = tier_header.replace(c, '')
    if '  ' in tier_header:
        tier_header = ' '.join(tier_header.split())    
    patterns_to_strip = re.findall(u'\"(.*?)\"', tier_header)    
    new_patterns = [p.replace(' ', '_') for p in patterns_to_strip]
    for old, new in zip(patterns_to_strip, new_patterns):
        tier_header = tier_header.replace(old, new)    
    
    
    tier_header = tier_header.strip().replace('ee T', 'ee_T').replace(' ', ',"')
    tier_header = '{"'+tier_header.replace('=', '":')+'}'
    return tier_header
    


def parse_tiers_old(elan_text):
    tiers_container = {}
    in_tier_bool = False
    current_tier = None
    for r in elan_text.split('\n'):
        if '<TIER' in r and in_tier_bool:            
            new_tier = parse_tier_header(r) 
            try:
                tiers_container[new_tier] = tiers_container[current_tier]   
            except:
                pass
            del tiers_container[current_tier]
            current_tier = new_tier

        elif '<TIER' in r and not in_tier_bool:
            current_tier = parse_tier_header(r)
            tiers_container[current_tier] = ''
            in_tier_bool = not in_tier_bool 
            
        elif '</TIER>' in r and in_tier_bool:

            tiers_container[current_tier]  = parse_annotations(tiers_container[current_tier])

            current_tier = None
            in_tier_bool = not in_tier_bool 
        
        elif in_tier_bool:
            tiers_container[current_tier] += r+'\n'

    # Final check to remove or process annotations without closed tiers
    to_delete = []
    for k in tiers_container.keys():
        if type(tiers_container[k]) == str:
            if '<ANNOTATION>' not in tiers_container[k]:
                to_delete.append(k)
            else:
                tiers_container[k]  = parse_annotations(tiers_container[k])
    for d in to_delete:
        del tiers_container[d]

    return tiers_container


def parse_tiers(elan_text):
    tiers_container = {}
    tier_headers = re.findall(r'\<TIER(.*?)\>', elan_text)
    # tier_headers = [t.strip() for t in tier_headers]
    t = tier_headers[0]
    for t in tier_headers:
        delimiter_1 = r'<TIER(.*?)>'.replace('(.*?)', t)
        tier_data = re.findall(delimiter_1+r'(.*?)</TIER>', elan_text, re.S)
        if len(tier_data) == 1:
            anns = parse_annotations(tier_data[0])
        elif len(tier_data) == 0:
            # print('empty TIER')
            continue
        elif len(tier_data) > 1:
            continue
            
        t += ' '
        vals = re.findall(r'\"(.*?)\"\s', t)
        for v in vals: t = t.replace(v, '')
        keys = re.findall(r'\s(.*?)\=', t)
        if 'TIER_ID' in keys:
            d = dict(zip(keys, vals))
            tiers_container[str(d)] = anns 

    return tiers_container



#%%############################################################################
'''                     FUNCTION parse media descriptor                     '''
###############################################################################
def parse_media_descriptor(elan_text):
    media_data = re.findall(r'\<MEDIA_DESCRIPTOR (.*?)\/\>', elan_text)
    media_data = [' '.join(m.split()) for m in media_data]
    
    preserving_data = [p for m in  media_data for p in re.findall(r'\"(.*?)\"', m)]
    replacer_tuple = [(p, p.replace(' ','$')) for p in  preserving_data]
    
    for old, new in replacer_tuple:
        elan_text = elan_text.replace(old, new)

    media_data = re.findall(r'\<MEDIA_DESCRIPTOR (.*?)\/\>', elan_text)
    media_data = [' '.join(m.split()) for m in media_data]
    media_data = ['{"'+m.strip().replace('=', '":').replace(' ',', "')+'}' for m in media_data]
    
    media_data_r = []
    for m in media_data:
        for old, new in replacer_tuple:
            m = m.replace(new, old)    
        media_data_r.append(m)
    try:
        media_data_r = [eval(m) for m in media_data_r]
    except SyntaxError:
        media_data_r = [m.replace('g"M', 'g", "M') for m in media_data_r]
        media_data_r = [eval(m) for m in media_data_r]
    return media_data_r

#%%############################################################################
'''                     FUNCTION correct timestamps                         '''
###############################################################################

def correct_timestamps(tier_container):
    tier_keys = tier_container.keys()
    
    ann_with_ts = []
    ann_without_ts = []
    for k in tier_keys:
        tier_content = tier_container[k]
        ann_with_ts += [{a : (tier_content[a][0], tier_content[a][1])} for a in tier_content
                           if tier_content[a][0] != '' and tier_content[a][1] != '']
        ann_without_ts += [{a : tier_content[a][-1]} for a in tier_content
                           if tier_content[a][0] == '' and tier_content[a][1] == '']
    
    mapper_ref_to_ts = {}
    for d in ann_with_ts:
        mapper_ref_to_ts.update(d)
        
    ann_without_ts_dict = {}
    for d in ann_without_ts:
        ann_without_ts_dict.update(d)
    mapped_ann = {k: mapper_ref_to_ts.get(v) for k, v in ann_without_ts_dict.items()}

    tier_container_corrected = tier_container
    for k in tier_keys:
        tier_content = tier_container[k]
        for ann_id in tier_content.keys():
            if ann_id in mapped_ann.keys():
                try:
                    tier_container_corrected[k][ann_id] = mapped_ann[ann_id]+tier_container_corrected[k][ann_id][2:]
                    
                except TypeError:
                    tier_container_corrected[k][ann_id] = ('missed','missed')+tier_container_corrected[k][ann_id][2:]
                    
    return tier_container_corrected

#%%############################################################################
'''                       FUNCTION map order timestamps                           '''
###############################################################################

def map_order_timestamps(tier_container, ts_mapper):
    new_tier_container = {}
    for k in tier_container.keys():
        old = tier_container[k]
        if len(old) > 0:
            new = []
            for ann_id in old:
                try:
                    new += [[ts_mapper[old[ann_id][0]], ts_mapper[old[ann_id][1]], old[ann_id][2]]]
                except KeyError:
                    new += [[-1, -1, old[ann_id][2]]]                    
            new = np.array(new)
            new_tier_container[k] = new[np.argsort(new[:, 0].astype(float))]
    return new_tier_container


#%%############################################################################
'''               FUNCTION create working directory function                '''
###############################################################################
def create_working_directory(top_path):
    subdirs = ['annotations', 'aligned_text']
    top_path.strip()
    if not os.path.exists(top_path):
        os.mkdir(top_path)
    
    handled_dirs = {}
    list_dir = os.listdir(top_path)
    for d in subdirs:
        if not d in list_dir:
            os.mkdir(top_path+'/'+d)
        handled_dirs[d] = top_path+'/'+d
    handled_dirs['top_dir'] = top_path
    return handled_dirs


#%%############################################################################
'''                        FUNCTION store single text                       '''
###############################################################################
def store_annotations(tier_container, output_path):
    
    participants = []
    tier_ids = []
    for k in tier_container.keys():
        tier_dict = eval(k)
        tier_keys = tier_dict.keys()
        # if tier_keys == ['LINGUISTIC_TYPE_REF']: continue
        if 'PARTICIPANT' in tier_keys:
            if 'TIER_ID' in tier_keys:
                participants.append(tier_dict['PARTICIPANT'])
                tier_ids.append(tier_dict['TIER_ID'])
        elif 'TIER_ID' in tier_keys:
            if tier_dict['TIER_ID'].lower() != 'example':
                participants.append('noparticipant')
                tier_ids.append(tier_dict['TIER_ID'])
        # else:
        #     assert False, 'NO HEADER FIELDS FOUND: Revise TIER params'
        # output_filename = ''.join([c for c in output_filename if c not i

    # CORRECTING PARTICIPANT PARAMETER
    tier_counts = [tier_ids.count(t) for t in tier_ids]
    if sum(tier_counts) == len(tier_counts):      # TIERS DETECTED ONLY FOR ONE PARTICIPANT
        if len(list(set(participants))) > 1:
            if 'noparticipant' in participants:
                actual_participants = [p for p in set(participants) if p != 'noparticipant']
                if len(actual_participants) == 1:
                    participants = [actual_participants[0] for i in range(len(tier_ids))]
                elif len(actual_participants) == 0:
                    participants = ['noparticipant' for i in range(len(tier_ids))]
                elif len(actual_participants) > 1:
                    participants = [actual_participants[0] for i in range(len(tier_ids))]
                    warnings.warn('Warning: Mismatched participants and tiers. {} selected as unique participant for output path: {}'.format(actual_participants[0],
                                                                                                                                             output_path))
                else:
                    assert False, 'Several participants found when it seems to be only one. Participants probably missed'
                
    for k, p, tier in zip(tier_container.keys(), participants, tier_ids):
        p = p if len(p) != 0 else 'noparticipant'
        tier_dict = eval(k)
        tier_keys = tier_dict.keys()

        tier = tier if not '/' in tier else tier.replace('/', '_')
        output_filename = '_'.join([p, tier])+'.txt'

        output_filename = ''.join([c for c in output_filename if c not in '&#;'])
        output_text = ''
        for r in tier_container[k]:
            output_text += ';'.join(r)+'\n'
        with open(os.path.join(output_path,output_filename), 'w', encoding = 'utf-8') as f:
            f.write(output_text)

#%%############################################################################
'''                      FUNCTION create alinged dataframe                  '''
###############################################################################
def align_text(txt_files, leading_modality):

    participant_list = [os.path.split(fn)[-1].split('_')[0] for fn in txt_files]

        
    uniq_participants = list(set(participant_list))
    output_dict = {}
    
    
    for p in uniq_participants:
        leading_txt_file = [fn for fn in txt_files if leading_modality in fn and p in fn]
        
        if len(leading_txt_file) == 0: continue   # NO LEADING ANNOTATIONS FOUND 
        leading_txt_file =leading_txt_file[0]
        
        with open(leading_txt_file, 'r', encoding = 'utf-8') as f:
            leading_anns = f.read()
        arr_leading = [r.split(';', maxsplit=2) for r in leading_anns.split('\n') if len(r.split(';')) > 2]
        
        if len(arr_leading) > 0:
            txt_no_leading = [fn for fn in txt_files if leading_modality not in fn and p in fn]
            ann_ids = [os.path.split(fn)[-1].split('_', maxsplit= 1)[-1].split('.')[0]
                         for fn in txt_no_leading]            
            no_leading_anns = []
            
            for n, fn in zip(ann_ids,txt_no_leading):
                with open(fn, 'r', encoding = 'utf-8') as f:
                    anns = f.read()            
                arr_anns = [r.split(';', maxsplit=2) for r in anns.split('\n') if len(r.split(';')) > 2]
                no_leading_anns.append((n, arr_anns))


            aligned_data = [['#'+n] for n in ann_ids]            
            
            for il, lead_row in enumerate(arr_leading):
                tstart = float(lead_row[0])
                tend = float(lead_row[1])
                for ik, (n, data) in enumerate(no_leading_anns):
                    data = np.array(data)
                    row_idx = [True if (float(row[0]) >= tstart and float(row[1]) <= tend) else False for row in data]
                    data_row = ['{}<{};{}>'.format(r[2], int(float(r[0])), int(float(r[1]))) if any(row_idx) else '#' for r in  data[row_idx, :]]
                    aligned_data[ik].append(' '.join(data_row))
                    
            aligned_data = [['#'+leading_modality]+['{}<{};{}>'.format(r[2], r[0], r[1]) for r in  arr_leading]] + aligned_data
            aligned_text = ['\n'.join(tlist) for tlist in aligned_data]
            
            p = p if len(p) != 0 else 'noparticipant' 
            output_dict[p]  = aligned_text             
    return output_dict

#%%############################################################################
'''                         FUNCTION parse elan file                        '''
###############################################################################

def parse_elan_file(elan_file : str, working_directory : str):
    with open(elan_file, 'r', encoding='utf-8') as f:
        elan_txt = str(f.read()).replace('>', '>\n').replace('\n\n', '\n')

    dir_handler = create_working_directory(working_directory)
    
    # CORRECTING ERRORS
    if '<TIER\n' in elan_txt:
        elan_txt = elan_txt.replace('<TIER\n', '<TIER')
    if '<TIME_SLOT\n' in elan_txt:
        elan_txt = elan_txt.replace('<TIME_SLOT\n', '<TIME_SLOT')
    if '<ALIGNABLE_ANNOTATION\n' in elan_txt:
        elan_txt = elan_txt.replace('<ALIGNABLE_ANNOTATION\n', '<ALIGNABLE_ANNOTATION')
    if '<MEDIA_DESCRIPTOR\n' in elan_txt:
        elan_txt = elan_txt.replace('<MEDIA_DESCRIPTOR\n', '<MEDIA_DESCRIPTOR')        
    if '"\n' in elan_txt:
        elan_txt = elan_txt.replace('"\n', '"')
    

    
    time_slots = parse_time_slots(elan_txt)
    tiers = parse_tiers(elan_txt)
    tiers_corrected = correct_timestamps(tiers)
    tiers_ordered = map_order_timestamps(tiers_corrected, time_slots)
    store_annotations(tiers_ordered, dir_handler['annotations'])
    media_data = parse_media_descriptor(elan_txt)
    return tiers_ordered, time_slots, media_data, dir_handler

#%%############################################################################
'''                       FUNCTION processing video                         '''
###############################################################################

def process_video(v_path, data_path, p):
    ''' PREPARING TIMESTAMPS  '''    
    modality_paths = [f for f in glob.glob(os.path.join(data_path,'annotations')+'/**/*', recursive = True)
                      if p in f]
    
    # FLUSH OLD MAPPED ANNOTATIONS
    modality_paths = [f if '_mapped.txt' not in f else os.remove(f) for f in modality_paths]
    modality_paths = [f for f in modality_paths if f is not None]

    if len(modality_paths) > 0:
        timestamps = []    
        for mf in modality_paths:
            with open(mf, 'r', encoding = 'utf-8') as f:
                modality_data = f.read()
            timestamps.append(np.array([[ts for ts in line.split(';')[:2]]
                                        for line in modality_data.split('\n')[:-1]]).astype(float))

                
    ''' VIDEO PROCESSING  '''
    vidcap = cv2.VideoCapture(v_path)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    time_slots = np.linspace(1/fps, duration+1/fps, frame_count)   
    output_videoname = p+'.mp4'
    output_path = os.path.join(data_path, output_videoname)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path,
                    fourcc,
                    int(fps), (frame_width, frame_height))

    frame_counter = 0
    stored_frame_counter = 0
    ts_frame_dict = {}
    ts_old_new_dict = {}
    # for t in time_slots:    
    while vidcap.isOpened():
        # t = np.round(t, decimals=2)*1e3
        success, image = vidcap.read()  
        if success:
            frame_counter += 1
            frame_ts = int(frame_counter/fps*1e3) # IN MILLISECONDS
            is_between_ranges = [True if frame_ts > tr[0] and frame_ts < tr[1] else False 
                                 for modality_ts in timestamps for tr in modality_ts]           
            if any(is_between_ranges):
                out.write(image)
                stored_frame_counter += 1
                ts_old_new_dict[frame_ts] = int(stored_frame_counter/fps*1e3)
        else: break
                    
    vidcap.release()
    out.release()
    # APPLY THE NEW TIMESTAMP TO ANNOTATION FILES
    apply_ts_mapper(data_path, p, ts_old_new_dict)
    
    
#%%############################################################################
'''                         PARSE ELAN FILES                                '''
############################################################################### 

def parse_elan_files(folder_with_elan_files,elan_file_list, working_dir):
    
    # CREATING THE OUTPUT FOLDER
    if working_dir not in os.listdir('./'):
        os.mkdir(working_dir)   
    
    progress_bar = tqdm(elan_file_list)
    with_error_parsing = {} # DICT TO KEEP THE FILES WITH ANY ERRORS
    detected_tier_id = []
    # STARTING THE PROCESSING
    for elan_filename, _ in zip(elan_file_list, progress_bar):
        output_filename = os.path.join(working_dir,
                                        os.path.split(elan_filename)[-1].split('.')[0].strip())
        if not folder_with_elan_files in elan_filename:
            elan_file_path = os.path.join(folder_with_elan_files, elan_filename)
        else:
            elan_file_path = elan_filename
    
        try:
            # EXTRACTING ALL INFORMATION FROM ELAN FORMAT
            tier_container, time_slots, media_data, _ = parse_elan_file(elan_file_path,
                                                           output_filename)    
            detected_tier_id += [eval(k)['TIER_ID'] for k in tier_container.keys() if 'TIER_ID' in k]
            detected_tier_id = list(set(detected_tier_id))
            # STORING RELEVANT DATA
            with open(os.path.join(output_filename, 'annontations_ts.dict'),
                      'w', encoding = 'utf-8') as f:
                f.write(str(time_slots))
            
            with open(os.path.join(output_filename, 'media_data.dict'),
                      'w', encoding = 'utf-8') as f:
                f.write(str(media_data))
                
        except IndexError as e:
            # IF INDEXERROR HAPPENS, THE SUBFOLDER IS FLUSHED AND THE ELAN ADDED 
            # IN THE PARSING REPORT
            with_error_parsing[elan_filename] = str(e)
            shutil.rmtree(output_filename)
    progress_bar.update(n=len(elan_file_list))
    with open(os.path.join(working_dir,'parsing_report.txt'),
              'w', encoding = 'utf-8') as f:
        f.write(str(with_error_parsing))
    return detected_tier_id

#%%############################################################################
'''                         ALIGN ANNOTATION                                '''
############################################################################### 

def align_annotations(working_dir : str, leading_modality : str):
    data_paths = os.listdir(working_dir)
    with_error_aligning = {} # DICT TO KEEP ERRORS
    
    progress_bar = tqdm(data_paths)
    for path, _ in zip(data_paths, progress_bar):
        if leading_modality is not None:
            ann_path = os.path.join(working_dir, path, 'annotations')
            try:
                txt_files = [os.path.join(ann_path, fn) for fn in os.listdir(ann_path) if '.txt' in fn] 
            except FileNotFoundError:
                with_error_aligning = (ann_path, 'No annotation folder found')
                continue
                
            if len(txt_files) > 0:
                has_leading = any([True if leading_modality in fn else False
                                for fn in txt_files])
                if has_leading:
                    text_dict = align_text(txt_files, leading_modality)               
                    for p in text_dict.keys():
                        for t in text_dict[p]:
                            txt_filename = '{}_{}.txt'.format(p, t.split('\n')[0].split('#')[1])
                            txt_filename = txt_filename if not '/' in txt_filename else txt_filename.replace('/', '_')
                            output_path = os.path.join(working_dir, path, 'aligned_text', txt_filename)
                            try:
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(t)
                            except:
                                with_error_aligning[ann_path] = 'Error storing aligned text'
                                break
                else:
                    with_error_aligning[ann_path] = 'Leading modality not found'
            else:
                with_error_aligning[ann_path] = ann_path, 'No annotations found'
        
    with open(os.path.join(working_dir, 'alignment_report.txt'),
              'w', encoding = 'utf-8') as f:
        f.write(str(with_error_aligning))       


#%%############################################################################
'''                       MERGING TEXT IN GLOBAL FILES                      '''
###############################################################################  

def merge_modalities(working_dir : str, required_modalities : list, elan_file_list : list):
    data_subdirs = [f for f in os.listdir(working_dir) if not '.txt' in f]
    txt_global_container = dict(zip([k for k in required_modalities]
                                          ,['' for _ in required_modalities]))
    
    for subdir in data_subdirs:
        text_files = [f for f in os.listdir(os.path.join(working_dir, subdir, 'aligned_text'))
                          if '.txt' in f]
        if len(text_files) > 0:
            participants = list(set([f.split('_')[0] for f in text_files]))
            
            txt_local_container = dict(zip(participants,[dict(zip([k for k in required_modalities]
                                              ,['' for _ in required_modalities])) for _ in participants]))
            
            # READING ANNOTATIONS
            for fn in text_files:
                p = fn.split('_')[0]
                fn = os.path.join(working_dir, subdir, 'aligned_text', fn)
                with open(fn, 'r', encoding = 'utf-8') as f:
                    text = f.read()
                fn_modality = [m for m in txt_local_container[p].keys() if m+'.txt' in fn][0]
                txt_local_container[p][fn_modality] = text
            # PADDING EMPTY MODALITIES
            for p in participants: 
                txt_lens = [len(txt_local_container[p][k].split('\n')) for k in txt_local_container[p].keys()] # These lens should equal
                for l,k in zip(txt_lens, txt_local_container[p].keys()):
                    if not l == max(txt_lens) and l == 1:
                        txt_local_container[p][k] = '#{}'.format(k)+'\n'*(max(txt_lens)-1)
            
            # ADDING TO GLOBAL CONTAINER
            for p in participants: 
                for k in txt_local_container[p].keys():
                    txt_global_container[k] += txt_local_container[p][k].replace('#{}'.format(k), '#{}-{}-{}'.format(k,p,subdir))+'\n'
    # STORING MERGED MODALITIES
    for k in txt_global_container.keys():
        with open(os.path.join(working_dir, k)+'.txt', 'w', encoding = 'UTF-8') as f:
            f.write(txt_global_container[k])
            
#%%############################################################################
'''                      PROCESSING VIDEOS - VERSION 1                      '''
# This version tries to match videonames with information contained in ELAN media field.
# This mode is unstable, but could be useful when it is difficult to identify participants.
###############################################################################
def extracting_visual_features_v1(top_video_path : str, working_dir : str):

    video_key = 'MEDIA_URL'
    video_extensions = ['mov', 'flv', 'mpg', 'mp4', 'avi', 'mpeg']
    
    
    video_paths = glob.glob(top_video_path+'/**/*', recursive = True)+glob.glob(top_video_path+'/**', recursive = True)
    video_paths = set([e for e in video_paths 
                    if  any([True if e.split('.')[-1].lower() == v_ext else False
                            for v_ext in video_extensions])])
    data_paths = [os.path.join(working_dir, i) for i in os.listdir(working_dir)
                  if os.path.isdir(os.path.join(working_dir, i))]
    no_proccessed = {}
    
    from difflib import SequenceMatcher
    import numpy as np
    
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    progress_bar = tqdm(data_paths)
    for path, _ in zip(data_paths, progress_bar):
        # READING MEDIA DATA
        with open(os.path.join(path, 'media_data.dict'), 'r', encoding = 'utf-8') as f:
            media_data = eval(f.read())
        videonames = [m[video_key].split('/')[-1] for m in media_data
                      if any([True if ex in m[video_key] else False for ex in video_extensions])]
        text_files = os.listdir(os.path.join(path, 'annotations'))
        participants = list(set([f.split('_')[0] for f in text_files]))
                
        # VIDEOS WITH DIFFERENT PARTICIPANTS
        video_mapping = zip(videonames,
                                  [[vp for vp in video_paths if v in vp] for v in videonames])
        videos_proccessed = 0
        for vn,vp in video_mapping:
            if len(vp) == 1:
                if len(participants) == 1:
                    p = participants[0]
                else:
                    p =  [pv for pv in participants if pv in vp[0]] #DETECTING PARTICIPANT
                    p = p[0] if len(p) == 1 else 'noparticipant'
                try:
                    process_video(vp[0], path, p)          
                except Exception as e:
                    no_proccessed[vp[0]] = str(e)
                    # raise e
                videos_proccessed += 1
                if len(participants) == videos_proccessed: break                      
            elif len(vp) > 1:
                # MATCHING BY NAME SIMILARITY WHEN SEVERAL VIDEOS DETECTED
                vn_similarity = [similar(vn, v) for v in vp]
                selected_v = np.argmax(vn_similarity)
                
                if len(participants) == 1:
                    p = participants[0]
                else:
                    p =  [pv for pv in participants if pv in vp[selected_v]] #DETECTING PARTICIPANT
                    p = p[0] if len(p) == 1 else 'noparticipant'
                try:
                    process_video(vp[selected_v], path, p)
                except Exception as e:
                    no_proccessed[vp[selected_v]] = str(e)
                videos_proccessed += 1
                if len(participants) == videos_proccessed: break
            elif len(vp) == 0:
                # VIDEO NOT FOUND AT FIRST STAGE. TRYING OTHER MATCHING WAYS
                vp2 = [v for v in video_paths if os.path.split(v)[-1].split('.')[0] in vn]
                if len(vp2) == 1:
                    
                    if len(participants) == 1:
                        p = participants[0]
                    else:
                        p =  [pv for pv in participants if pv in vp2[0]] #DETECTING PARTICIPANT
                        p = p[0] if len(p) == 1 else 'noparticipant'
                    try:
                        process_video(vp2[0], path, p)
                    except Exception as e:
                        no_proccessed[vp2[0]] = str(e)        
                    videos_proccessed += 1
                    if len(participants) == videos_proccessed: break
                elif len(vp2) > 1:
                    vn_similarity = [similar(vn, v) for v in vp2]
                    selected_v = np.argmax(vn_similarity)
                    
                    if len(participants) == 1:
                        p = participants[0]
                    else:
                        p =  [pv for pv in participants if pv in vp2[selected_v]] #DETECTING PARTICIPANT
                        p = p[0] if len(p) == 1 else 'noparticipant'
                    try:
                        process_video(vp2[selected_v], path, p)
                    except Exception as e:
                        no_proccessed[vp2[0]] = str(e)
                    videos_proccessed += 1
                    if len(participants) == videos_proccessed: break
                elif len(vp2) == 0:
                    no_proccessed[vn] = 'No matching found'      
     
    with open('no_proccessed.txt', 'w', encoding = 'utf-8') as f:
        f.write(str(no_proccessed))    
    
    

#%%############################################################################
'''                      PROCESSING VIDEOS - VERSION 2                      '''
# This version works when there is matching between participants and videonames inside elan files.
# This mode is fast and safe when participant names are clearly identified in elan fields.
###############################################################################
def extracting_visual_features_v2(top_video_path : str, working_dir : str):
    video_extensions = ['mov', 'flv', 'mpg', 'mp4', 'avi', 'mpeg']
    video_paths = glob.glob(top_video_path+'/**/*', recursive = True)+glob.glob(top_video_path+'/**', recursive = True)
    video_paths = set([e for e in video_paths 
                    if  any([True if e.split('.')[-1].lower() == v_ext else False
                            for v_ext in video_extensions])])
    
    
    no_proccessed = {} # DICT TO KEEP ERRORS
    media_files = glob.glob(working_dir+'/**/*/media_data.dict', recursive = True)
    progress_bar = tqdm(media_files)
    for m, _ in zip(media_files, progress_bar):
        # DETECTING PARTICIPANTS
        participants = list(set([fp.split('_')[0]
                for fp in os.listdir(os.path.join(os.path.split(m)[0], 'annotations'))]))
        # READING MEDIA DATA
        with open(m, 'r', encoding = 'UTF-8') as f:
            media_data = eval(f.read())
        
        for media_element in media_data:
            media_element = media_element['MEDIA_URL']
            vn_in_elan = os.path.split(media_element)[-1]
            
            # SEEKING A MATCHING VIDEO
            matched_vn = [v for v in video_paths if vn_in_elan in v]
            if len(matched_vn) == 1: # VIDEO FOUND IN THE VIDEO FOLDER
                matched_vn = matched_vn[0]
                # SEEKING A MATCHING PARTICIPANT
                sel_participant = [p for p in participants if p in matched_vn]
                if len(sel_participant) == 1: # PARTICIPANT CORRECTLY DETECTED
                    data_path = os.path.split(m)[0]
                    process_video(matched_vn, data_path, sel_participant[0])
                elif len(matched_vn) == 0:
                    no_proccessed[vn_in_elan] = 'No participant match'     
                    
            elif len(matched_vn) == 0:
                no_proccessed[vn_in_elan] = 'No matching elan folder'
        
    with open(os.path.join(working_dir, 'video_processing.txt'), 'w',
              encoding = 'utf-8') as f:
        f.write(str(no_proccessed))

#%%############################################################################
'''                      PROCESSING VIDEOS - VERSION 3                      '''
# This version works when there is matching between participants and videonames outside the elan files.
# This mode is fast and safe when participant names are clearly identified in the videonames.
###############################################################################
def extracting_visual_features_v3(top_video_path  : str, working_dir : str):
    video_extensions = ['mov', 'flv', 'mpg', 'mp4', 'avi', 'mpeg']
    video_paths = glob.glob(top_video_path+'/**/*', recursive = True)+glob.glob(top_video_path+'/**', recursive = True)
    video_paths = set([e for e in video_paths 
                    if  any([True if e.split('.')[-1].lower() == v_ext else False
                            for v_ext in video_extensions])])

    no_proccessed = {} # DICT TO KEEP ERRORS
    data_paths = [f for f in os.listdir(working_dir)
                    if os.path.isdir(os.path.join(working_dir,f))]
    
    progress_bar = tqdm(data_paths)
    for dp, _ in zip(data_paths, progress_bar):
        # DETECTING PARTICIPANTS
        participants = list(set([fp.split('_')[0]
                for fp in os.listdir(os.path.join(working_dir, dp, 'annotations'))]))
        presel_vns = [vn for vn in video_paths if dp in vn]
        if len(presel_vns) > 0:
            for p in participants:
                matching_vn = [vn for vn in presel_vns if p in vn]
                if len(matching_vn) == 1:
                    data_path = os.path.join(working_dir, dp)
                    process_video(matching_vn[0], data_path, p)
                else:
                    no_proccessed[dp] = 'None matching video'
        else:
            no_proccessed[dp] = 'None matching video'
            
    with open(os.path.join(working_dir, 'video_processing.txt'), 'w',
              encoding = 'utf-8') as f:
        f.write(str(no_proccessed))

#%%############################################################################
'''                            APPLY TS MAPPER                              '''
###############################################################################

map_to_int = lambda x : int(x.split('.')[0])
map_to_str_ts = lambda x : str(x)+'.0'

def get_nearest_ts(ts,d):
    dist = [(ts-t)**2 for t in d.keys()]
    nearest = [x for _, x in sorted(zip(dist, d.keys()))]
    return nearest[0]
    
def replace_timestamp(s, d):
    ori_ts = list(map(map_to_int, s.split(';')[:2]))
    new_ts = [str(d[ot]) if ot in d.keys() else \
               str(d[get_nearest_ts(ot,d)])  for ot in ori_ts]
    # CREATING THE PATTERN FOR REGULAR EXPRESSION
    d_s = dict(zip(map(map_to_str_ts,ori_ts), map(map_to_str_ts,new_ts)))
    p = "|".join(d_s.keys())
    p = r"(?<!\w)(" + p + r")(?!\w)"
    return re.compile(p).sub(lambda m: d_s[m.group(0)], s)


def apply_ts_mapper(data_path, participant, ts_mapper):
    ann_path = os.path.join(data_path, 'annotations')

    ann_files = [os.path.join(ann_path, m)
                  for m in os.listdir(ann_path) if '.txt' in m and not '_mapped.txt' in m and participant in m]
    # ts_mapper = dict(zip([str(k*1000.0) for k in ts_mapper.keys()],
    #                 [str(v*1000.0) for v in ts_mapper.values()]))   
    for fn in ann_files:
        with open(fn, 'r', encoding = 'utf-8') as f:
            annotations = f.read()
        ann_mapped = [replace_timestamp(s, ts_mapper) for s in annotations.split('\n') if len(s) > 0]
        ann_mapped = '\n'.join(ann_mapped)
        output_path = fn.replace('.txt', '_mapped.txt')
        with open(output_path, 'w', encoding = 'utf-8') as f:
            f.write(ann_mapped)
            


