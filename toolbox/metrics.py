import pretty_midi
import numpy as np
import sys, os
import midi
import glob
import math
# metircs
class metrics(object):

    
    def total_used_pitch(self, piano_roll): #return a tuple
        sum_notes = np.sum(piano_roll, axis=1)
        return np.sum(sum_notes>0)

    def bar_used_pitch(self, pattern, track_num=0,num_bar = False): #return [num_bar,used_pitch]
    
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        for i in range(0,len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[0]*resolution*4/ 2**(time_sig[1])
                if num_bar == False:
                    num_bar = int(round(float(pattern[track_num][-1].tick)/bar_length))
                    used_notes = np.zeros((num_bar,1))
                else:
                    used_notes = np.zeros((num_bar,1))
                
            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] !=0:
                if 'time_sig' not in locals(): #set default bar length as 4 beat
                    bar_length = 4*resolution
                    time_sig = [4, 2, 24, 8]
                    
                    if num_bar == False:
                        num_bar = int(round(float(pattern[track_num][-1].tick)/bar_length))
                        used_notes = np.zeros((num_bar,1))     
                        used_notes[pattern[track_num][i].tick/bar_length] += 1
                    else:
                        used_notes = np.zeros((num_bar,1))     
                        used_notes[pattern[track_num][i].tick/bar_length] += 1
                    note_list = []
                    note_list.append(pattern[track_num][i].data[0])  
                    

                else:
                    for j in range(0, num_bar):
                        if 'note_list'in locals():
                            pass
                        else:
                            note_list = []
                    note_list.append(pattern[track_num][i].data[0]) 
                    used_notes[pattern[track_num][i].tick/bar_length] += 1

        used_pitch = np.zeros((num_bar,1))
        current_note = 0
        for i in range(0,num_bar):
            used_pitch[i] = len(set(note_list[current_note:current_note+int(used_notes[i][0])]))
            current_note += int(used_notes[i][0])

        return used_pitch



    def total_used_note(self, pattern, track_num=0): #return a tuple
        notes = 0
        for i in range(0,len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] !=0:
                notes +=1
        return notes

    def bar_used_note(self, pattern, track_num=0, num_bar = False): #return shape [num_bar,used_notes]
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        for i in range(0,len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[track_num]*resolution*4/ 2**(time_sig[1])
                if num_bar == False:
                    num_bar = int(round(float(pattern[track_num][-1].tick)/bar_length))
                    used_notes = np.zeros((num_bar,1))
                else:
                    used_notes = np.zeros((num_bar,1))
                
                
            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] !=0:
                if 'time_sig' not in locals(): #set default bar length as 4 beat
                    bar_length = 4*resolution
                    time_sig = [4, 2, 24, 8]

                    if num_bar == False:
                        num_bar = int(round(float(pattern[track_num][-1].tick)/bar_length))
                        used_notes = np.zeros((num_bar,1))
                        used_notes[pattern[track_num][i].tick/bar_length] += 1
                    else:
                        used_notes = np.zeros((num_bar,1))
                        used_notes[pattern[track_num][i].tick/bar_length] += 1
                    
                else:
                    used_notes[pattern[track_num][i].tick/bar_length] += 1
        return used_notes


    def total_pitch_class_histogram(self, piano_roll):  #return histrogram of 12 pitch, with weighted duration shape 12
        histogram = np.zeros(12)
        for i in range(0,128):
            pitch_class = i%12
            histogram[pitch_class] += np.sum(piano_roll, axis =1)[i]

        return histogram/sum(histogram)

    def bar_pitch_class_histogram(self, pm_object, bpm=120, num_bar = False, track_num=0): #return shape [num_bar, 12]
        #todo: deal with more than one time signature cases
        if num_bar == False:
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_length = 60./bpm*numer*4/deno*100
            piano_roll = pm_object.instruments[track_num].get_piano_roll(fs=100)
            piano_roll = np.transpose(piano_roll,(1,0))
            actual_bar = len(piano_roll)/bar_length
            num_bar = int(round(actual_bar))
            bar_length = int(round(bar_length))
        else:
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_length = 60./bpm*numer*4/deno*100
            piano_roll = pm_object.instruments[track_num].get_piano_roll(fs=100)
            piano_roll = np.transpose(piano_roll,(1,0))
            actual_bar = len(piano_roll)/bar_length
            bar_length = int(math.ceil(bar_length))
        
        if actual_bar > num_bar:
            piano_roll = piano_roll[:-np.mod(len(piano_roll),bar_length)].reshape((num_bar,-1,128))     #make exact bar 
        elif actual_bar == num_bar:
            piano_roll = piano_roll.reshape((num_bar,-1,128)) 
        else:
            #print piano_roll.shape
            #print num_bar*bar_length,len(piano_roll)
        
            piano_roll = np.pad(piano_roll,((0,int(num_bar*bar_length-len(piano_roll))),(0,0)), mode='constant', constant_values=0)
            piano_roll = piano_roll.reshape((num_bar,-1,128)) 
        
        bar_histogram = np.zeros((num_bar,12))
        for i in range(0, num_bar):
            histogram = np.zeros(12)
            for j in range(0,128):
                pitch_class = j%12
                histogram[pitch_class] += np.sum(piano_roll[i], axis =0)[j]
            if sum(histogram)!=0:
                bar_histogram[i] = histogram/sum(histogram)
            else:
                bar_histogram[i] = np.zeros(12)
        return bar_histogram

    def pitch_class_transition_matrix(self, pm_object, normalize = 0): #return shape [12, 12]
        transition_matrix = pm_object.get_pitch_class_transition_matrix()

        if normalize == 0:
            return transition_matrix
        
        elif normalize == 1:
            
            sums = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1,1)

        elif normalize == 2:
            
            return transition_matrix/sum(sum(transition_matrix))
        
        else:
            print "invalid normalization mode, return unnormalized matrix"
            return transition_matrix

    def pitch_range(self, piano_roll):
        used_pitch = np.sum(piano_roll, axis=1)>0
        pitch_index = np.where(used_pitch==True)
        return np.max(pitch_index) - np.min(pitch_index)
        
           
    def chord_dependency(self, pm_object, bar_chord,bpm=120, num_bar = False, track_num=0): #return tuple
        #compare bar chroma with chord chroma. calculate the ecludian
        bar_pitch_class_histogram = self.bar_pitch_class_histogram(pm_object,bpm=bpm,num_bar = num_bar, track_num=track_num)
        dist = np.zeros((len(bar_pitch_class_histogram)))
        for i in range((len(bar_pitch_class_histogram))):
            dist[i] = np.linalg.norm(bar_pitch_class_histogram[i]-bar_chord[i])
        average_dist = np.mean(dist)
        return average_dist
                       
    def avg_pitch_shift(self, pattern, track_num=0):  #return tuple
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        total_used_note = self.total_used_note(pattern, track_num=track_num)
        d_note = np.zeros((total_used_note-1))
        current_note = 0
        counter = 0
        for i in range(0,len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] !=0:
                
                if counter != 0:
                    d_note[counter-1] = current_note-pattern[track_num][i].data[0]
                    current_note = pattern[track_num][i].data[0]
                    counter+=1
                else:
                    current_note = pattern[track_num][i].data[0]
                    counter+=1
        return np.mean(abs(d_note))
    def avg_IOI(self, pm_object):
        onset = pm_object.get_onsets()
        ioi = np.diff(onset)
        return np.mean(ioi)
    #define: [full, half, quarter, 8th, 16th, dot_half, dot_quarter, dot_8th, dot_16th, half note triplet, quarter note triplet, 8th note triplet, others]
    def note_length_hist(self, pattern, track_num = 0, normalize = True, pause_event = False):
        if pause_event == False:
            note_length_hist = np.zeros((12))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            #basic unit: bar_length/96
            for i in range(0,len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num]*resolution*4/ 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] !=0:
                    if 'time_sig' not in locals(): #set default bar length as 4 beat
                        bar_length = 4*resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length/96.
                    #tol = 6.*unit
                    hist_list = [unit*96, unit*48, unit*24, unit*12, unit*6, unit*72, unit*36, unit*18, unit*9, unit*32, unit*16, unit*8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    #find next note off
                    for j in range(i,len(pattern[track_num])):
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] ==0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - current_tick
                                distance = np.abs(np.array(hist_list)-note_length)
                                idx = distance.argmin()
                                #if distance[idx] > tol:
                                #    note_length_hist[-1]+=1
                                #else:
                                note_length_hist[idx]+=1
                                break
        else:
            note_length_hist = np.zeros((24))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            #basic unit: bar_length/96
            
            for i in range(0,len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num]*resolution*4/ 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] !=0:
                    check_previous_off = True
                    if 'time_sig' not in locals(): #set default bar length as 4 beat
                        bar_length = 4*resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length/96.
                    tol = 3.*unit
                    hist_list = [unit*96, unit*48, unit*24, unit*12, unit*6, unit*72, unit*36, unit*18, unit*9, unit*32, unit*16, unit*8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    #find next note off
                    for j in range(i,len(pattern[track_num])):
                        #find next note off
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] ==0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - current_tick
                                distance = np.abs(np.array(hist_list)-note_length)
                                idx = distance.argmin()
                                note_length_hist[idx]+=1
                                break
                            else:
                                if pattern[track_num][j].tick == current_tick:
                                    check_previous_off = False
                        #find note off
                    #find previous note off/on
                    if check_previous_off == True:
                        for j in range(i-1,0,-1):
                            if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] !=0:
                                break

                            elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] ==0):

                                note_length =  current_tick - pattern[track_num][j].tick
                                distance = np.abs(np.array(hist_list)-note_length)
                                idx = distance.argmin()
                                if distance[idx] < tol:
                                    note_length_hist[idx+12]+=1
                                break
            
        if normalize == False:
            return note_length_hist

        elif normalize == True:
            
            return note_length_hist / np.sum(note_length_hist)
        

    def note_length_transition_matrix(self, pattern, track_num=0, normalize = 0, pause_event = False):
        if pause_event == False:
            transition_matrix = np.zeros((12,12))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            idx = None
            #basic unit: bar_length/96
            for i in range(0,len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num]*resolution*4/ 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] !=0:
                    if 'time_sig' not in locals(): #set default bar length as 4 beat
                        bar_length = 4*resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length/96.
                    hist_list = [unit*96, unit*48, unit*24, unit*12, unit*6, unit*72, unit*36, unit*18, unit*9, unit*32, unit*16, unit*8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    #find note off
                    for j in range(i,len(pattern[track_num])):
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] ==0):
                            if pattern[track_num][j].data[0] == current_note:
                                note_length = pattern[track_num][j].tick - current_tick
                                distance = np.abs(np.array(hist_list)-note_length)

                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx != None:
                                    transition_matrix[last_idx][idx]+=1
                                break
        else: 
            transition_matrix = np.zeros((24,24))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            idx = None
            #basic unit: bar_length/96
            for i in range(0,len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num]*resolution*4/ 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] !=0:
                    check_previous_off = True
                    if 'time_sig' not in locals(): #set default bar length as 4 beat
                        bar_length = 4*resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length/96.
                    tol = 3.*unit
                    hist_list = [unit*96, unit*48, unit*24, unit*12, unit*6, unit*72, unit*36, unit*18, unit*9, unit*32, unit*16, unit*8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    #find next note off
                    for j in range(i,len(pattern[track_num])):
                        #find next note off
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] ==0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - current_tick
                                distance = np.abs(np.array(hist_list)-note_length)
                                
                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx != None:
                                    transition_matrix[last_idx][idx]+=1
                                break
                            else:
                                if pattern[track_num][j].tick == current_tick:
                                    check_previous_off = False
                        #find note off
                    #find previous note off/on
                    if check_previous_off == True:
                        for j in range(i-1,0,-1):
                            if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] !=0:
                                break

                            elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] ==0):

                                note_length =  current_tick - pattern[track_num][j].tick
                                distance = np.abs(np.array(hist_list)-note_length)
                                
                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx != None :
                                    if  distance[idx] < tol:
                                        idx = last_idx
                                        transition_matrix[last_idx][idx+12]+=1
                                break
                                
        if normalize == 0:
            return transition_matrix

        elif normalize == 1:
            
            sums = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1,1)

        elif normalize == 2:
            
            return transition_matrix/sum(sum(transition_matrix))
        
        else:
            print "invalid normalization mode, return unnormalized matrix"
            return transition_matrix
        