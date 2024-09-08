#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:23:55 2022

@author: imac
"""

#Code for Caroline
from scipy.stats import linregress
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
import os
import csv

stim_preset_times = False
stim_peak = -0.4 #millivolts # variable
stim_slope_dt = 1 #[samples], the increment in samples over which to check stim slope. Default = 1
stim_slope_thresh = 0.3 # variable
fs = 20000 #Sampling frequency of data [Hz]
epsp_highcut = 500 #[Hz] cutoff freq of EPSP LPF. ** 500Hz was original value.
n_channels = 2

# def load_presets(filename):
#     with open(filename+'.csv', newline='') as f:
#         lines = csv.reader(f)
#     data = list(lines)
#     return data

# presets = load_presets("presets")
presets = [
    [1.005, 1.205, 1.405, 1.605, 1.805, 2.005, 2.205, 2.405, 2.605, 2.805],
[1.005, 1.055, 1.105, 1.155, 1.205, 1.255, 1.305, 1.355, 1.405, 1.455],
[1.005, 1.025, 1.045, 1.065, 1.085, 1.105, 1.125, 1.145, 1.165, 1.185],
[1.005, 1.025, 1.045, 1.265, 1.285, 1.305, 1.525, 1.545, 1.565, 1.785, 1.805, 1.825, 2.045, 2.065, 2.085],
[1.005, 1.045, 1.085, 1.125, 1.165, 1.205, 1.245, 1.285, 1.325, 1.365],
[1.005, 1.025, 1.045, 1.245, 1.265, 1.285, 1.485, 1.505, 1.525, 1.725, 1.745, 1.765, 1.965, 1.985, 2.005]
]

def sample_to_sec(samples): #converts a sample number to absolute time in seconds
    secs = samples/fs
    return secs 

def low_pass_filter2(data, fc, fs, order): #no phase shift
    nyq = 0.5 * fs
    cutoff = fc/nyq
    #scipy.signal.butter(9, Wn, btype='low', analog=False, output='ba', fs=None) #returns two items, 
    b, a = butter(order, cutoff, btype='lowpass', analog=False, output='ba', fs=None) #returns critical bounds to be used for actual filter , 
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def remove_slow_wave(sig): #Function for removing slow-wave prior to spike detection
    sig_filtered = low_pass_filter2(sig, epsp_highcut, fs, order=2)
    sig_return = sig - sig_filtered
    return sig_return


def stim_train_detect(sig, stim_cut, stim_slope_thresh):     #returns list of spike starts and stops, in samples.
    #print("\nDetecting stim train")
    #start_time = time.time()
    stim_starts = []
    stim_stops = []
    i = 0
    siglength = len(sig)
    # print("stim_train_detect: sig length",siglength)
    # print("sig shape",np.shape(sig))
    # print("sig sample",sig[0:10])
    time_intervals_s = []
 
    #=======================================================================
    if stim_preset_times == False: # unknown time intervals between spikes (stim = spike)
        stimtrain_overthresh_bool = [0] * siglength
        is_included = "no"
        if stim_peak > 0: 
            stim_deflection = "up"
        else: 
            stim_deflection = "down"
        # ======= Upward deflecting stim ====== At first sample over threshold, looks back for slope, if steep enough will count as stim spike.
        if stim_deflection == "up":
            for i in (range(siglength)): # i = [samples]
                  if float(sig[i]) >= float(stim_cut): 
                    stimtrain_overthresh_bool[i] = 1  # upward deflection
                  else: 
                    stimtrain_overthresh_bool[i] = 0
                  if i > 0:
                      if (stimtrain_overthresh_bool[i]) > (stimtrain_overthresh_bool[i-1]): 
                         #check slope before adding start
                         
                         #10/5/21 need to find local max before defining falling slope....
                         
                         slope_left, intercept, r, p, se = linregress([i-stim_slope_dt, i], [sig[i-stim_slope_dt],sig[i]]) # looking at the relationship between per entry the signal values
                         #slope_right, intercept, r, r, se = linregress([i, i + stim_slope_dt], [sig[i],sig[i + stim_slope_dt]])   #<--------- not going to work as-is.
    
                         #print("candidate left slope", slope_left, "at ", sample_to_sec(i), "seconds")
                         #print("candidate right slope", slope_right, "at ", sample_to_sec(i), "seconds")
                         
                         #if slope_left >= stim_slope_thresh and slope_right <= -stim_slope_thresh:
                         if slope_left >= stim_slope_thresh:
                             stim_starts.append(i)
                             print("Stim artifact rising edge detected, slope ",round(slope_left,4), "at ",sample_to_sec(i), "secs")
                             is_included = "yes"
                             time_intervals_s.append(sample_to_sec(i))
                         else:
                             is_included = "no"
                      if stimtrain_overthresh_bool[i-1] > stimtrain_overthresh_bool[i]:
                         if is_included == "yes":
                             #print("stim stop at ", sample_to_sec(i))
                             #if slope <= -stim_slope_thresh:
                             stim_stops.append(i)
        else: #downward deflection. Still looks for rising edge
            # print("DEBUG: Stim spike deflection down")
            for i in (range(siglength)): # i = [samples]
                  if float(sig[i]) <= float(stim_cut): 
                    stimtrain_overthresh_bool[i] = 1  # upward deflection
                  else: 
                    stimtrain_overthresh_bool[i] = 0
                  if i > 0:
                      if (stimtrain_overthresh_bool[i]) > (stimtrain_overthresh_bool[i-1]): # only when 0 followed by 1 -- greater than threshold --> less than threshold
                         #check slope before adding start
                         slope, intercept, r, p, se = linregress([i-stim_slope_dt, i], [sig[i-stim_slope_dt],sig[i]]) # slope between two points
                        #  print("candidate starts slope", slope, "at ", sample_to_sec(i), "seconds")
                         if slope <= -stim_slope_thresh: # stim_slope_thresh is positive, absolute slope between two points must be greater than threshold
                             stim_starts.append(i)
                            #  print("*Stim artifact falling edge detected, slope ",round(slope,2), "at ",sample_to_sec(i), "secs")
                             is_included = "yes"
                             time_intervals_s.append(sample_to_sec(i))
                         else:
                             is_included = "no"
                      if stimtrain_overthresh_bool[i-1] > stimtrain_overthresh_bool[i]:
                         if is_included == "yes":
                            #  print("stim stop at ", sample_to_sec(i))
                             #if slope <= -stim_slope_thresh:
                             stim_stops.append(i)
                

        starts_stops_length = int(len(stim_starts))
        stim_spike_widths = []
        for k in range(0,starts_stops_length-1): 
            stim_spike_widths = np.append(stim_spike_widths, stim_stops[k] - stim_starts[k])
        stim_n_spikes = len(stim_starts)
        #print("number of stim spikes detected",stim_n_spikes, ", at locations", stim_starts)
        #end_time = time.time()
        #time_elapsed = end_time - start_time
        #print("time elapsed detecting stim spikes",time_elapsed)
        #print("stim starts",stim_starts)
        #print("stim stops",stim_stops)        
    return(stim_starts, stim_stops, stim_spike_widths, stim_n_spikes, time_intervals_s)     


def arr_matched(arr1, arr2): # computes the percentage of patterns matched across arrays
    """
    arr1 = [1.001, 1.021, 1.041, 1.241, 1.481, 1.501, 1.521, 1.721, 1.741, 1.761, 1.961, 1.981, 2.001]
    arr2 = [1.005, 1.025, 1.045, 1.245, 1.265, 1.285, 1.485, 1.505, 1.525, 1.725, 1.745, 1.765, 1.965, 1.985, 2.005]
    """
    num_matched = 0
    diff = round(arr1[0] - arr2[0], 5)
    arr_len_diff = -abs(len(arr1) - len(arr2))
    offset = 0

    if len(arr1) >= len(arr2):
        wide_arr = arr1
        small_arr = arr2
    else:
        wide_arr = arr2
        small_arr = arr1

    for i in range(len(wide_arr)):
        if len(small_arr) == i + offset:
            break
        if diff == round(wide_arr[i] - small_arr[i + offset], 5):
            num_matched += 1
        else:
            offset = max(offset - 1, arr_len_diff)
    return num_matched / len(wide_arr)

def preset_matching(arr): # computes the preset matching
    percentage_matched = []
    for preset in presets:
        percentage_matched.append(arr_matched(preset, arr))
    return percentage_matched

def highest_preset(arr):
    """
    [1,2,3,4,1]
    max_val = max(presets_matched)
    for i in range(len(presets_matched)):
        if presets_matched[i] == max_val:
            return i
    """
    presets_matched = preset_matching(arr) # array of percentages
    return presets_matched.index(max(presets_matched))



# ==== Run program =============
directory = ""
dir_list = os.listdir(".")
print("Text files in this directory:")
for file_name in dir_list:
    if ('.txt' in file_name[-4:]):
        print(file_name)
        file = file_name
print("====================")
print("File to be processed:")
print(file)
print("Importing file, please wait.... ")

f = open(file, 'r')
data = np.empty(1)
data = [x.strip().split('\t') for x in f]
f.close() 
data_0 = [item[0] for item in data] #gets rid of those annoying quotations marks that each data item has when imported.
data_float = [float(x) if x!='' else 0 for x in data_0] #if gap in data, replace with a 0.
del data_0
# ============================
# print("file data beginning snippet",data_float[0:10])
gap_list = [(index) for index, element in enumerate(data) if element == ['']] #generates list of indices for starts of trials 
gap_list = ([-1] + gap_list)   
n_trials = len(gap_list)-1 
n_trials = len(gap_list) 
del data
print("data loaded..")
trial_numbers = range(len(gap_list)-1)


df_cols = ["Trial Number", "Stim Peak", "Stim Slope Threshold", "Matched Preset", "Stim Spikes", "Time Intervals"]
final_data_df = pd.DataFrame(columns = df_cols)

for i in (range(len(trial_numbers))):
    trial_number = trial_numbers[i]       
    #One graph per trial. 
    #print("------------ NEW TRIAL ----------------")
    #=======  Generate trial number  ============:
    if n_channels == 2:
        if (divmod(trial_number, 2))[1] == 0:
            channel = 1
        else: channel = 2
    else: channel = 1
    print("\n\nProcessing trial", trial_number)
                    
    #====== Pick trial to assign to master_data  ========= 
    #print("gap list", gap_list, "length", len(gap_list))
    if len(gap_list) > 1:
        if n_channels == 2:
            trial_data = data_float[(gap_list[trial_number]+1):(gap_list[trial_number+1])]
        elif n_channels == 1:
            trial_data = data_float[(gap_list[trial_number]+1):(gap_list[trial_number+1])]
            #trial_data = data_float[(gap_list[trial_number-1]):(gap_list[trial_number])]
    #except:
        #print("*** ERROR trial numbers out of range: Most likely data file does not contain specified number of pre/post trials.")
        #print("gap list indices for DATA channel: ", (gap_list[trial_number]+1), (gap_list[trial_number+1]))
    else: #single trial mode.
        trial_data = data_float
    master_data = np.array(trial_data)
    del trial_data

    stim_data_filt = remove_slow_wave(master_data) 
    

    stim_peaks = np.linspace(-0.5, -0.3,num=5) # -0.5 <= n <= -0.3
    stim_slope_thresholds = np.linspace(0.2, 0.4, num=5)

    
    for stim_peak in np.nditer(stim_peaks):
        for stim_slope_threshold in np.nditer(stim_slope_thresholds):
            # print("Stim Peak:", stim_peak, " | Slope Threshold:", stim_slope_threshold)
            stim_starts, stim_stops, stim_widths, stim_n_spikes, time_intervals = stim_train_detect(stim_data_filt, stim_peak, stim_slope_threshold)

            matched_preset = []
            if time_intervals != []:
                matched_preset_index = highest_preset(time_intervals)
                matched_preset = presets[matched_preset_index]
                print("Potential Preset Match:", matched_preset, "\n\n")
                
            
            # print(stim_peak)

            data_dict = {
                "Trial Number": trial_number, 
                "Stim Peak": stim_peak,
                "Stim Slope Threshold": stim_slope_threshold,
                "Matched Preset": [matched_preset],
                "Stim Spikes": stim_n_spikes, 
                "Time Intervals": [time_intervals]
            }

            # print(data_dict)
            # row_data = pd.Series([trial_number, stim_peak, stim_slope_threshold, matched_preset, stim_n_spikes, time_intervals])
            # print(row_data.transpose())
            
            new_df = pd.DataFrame.from_dict(data_dict)
            # print(new_df.head())

            final_data_df = pd.concat([final_data_df, new_df], axis = 0)
    # stim_starts, stim_stops, stim_widths, stim_n_spikes, time_intervals = stim_train_detect(stim_data_filt, stim_peak, stim_slope_thresh) 
    
final_data_df = final_data_df.reset_index(drop=True)
final_data_df.to_csv("stim_analysis.csv")

    # print(stim_starts)
    # print(stim_stop

    # for i in range(len(stim_starts)):
    #     print("Start: ", stim_starts[i])
    #     print("Stop: ", stim_stops[i], "\n")

# print out highest percentage match and compare with output
# graph outputs and layer over  original graphs


# Columns: Trial Number, slope thresh, stim peak, which preset, list of amplitude of the artifacts, offset value
# 
