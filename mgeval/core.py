# coding:utf-8
"""core.py
Include feature extractor and musically informed objective measures.
"""
import pretty_midi
import numpy as np
import sys
import os
import midi
import glob
import math


# feature extractor
def extract_feature(_file):
    """
    This function extracts two midi feature:
    pretty_midi object: https://github.com/craffel/pretty-midi
    midi_pattern: https://github.com/vishnubob/python-midi

    Returns:
        dict(pretty_midi: pretty_midi object,
             midi_pattern: midi pattern contains a list of tracks)
    """
    feature = {'pretty_midi': pretty_midi.PrettyMIDI(_file),
               'midi_pattern': midi.read_midifile(_file)}
    return feature


# musically informed objective measures.
class metrics(object):
    def total_used_pitch(self, feature):
        """
        total_used_pitch (Pitch count): The number of different pitches within a sample.

        Returns:
        'used_pitch': pitch count, scalar for each sample.
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        sum_notes = np.sum(piano_roll, axis=1)
        used_pitch = np.sum(sum_notes > 0)
        return used_pitch

    def bar_used_pitch(self, feature, track_num=1, num_bar=None):
        """
        bar_used_pitch (Pitch count per bar)

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.

        Returns:
        'used_pitch': with shape of [num_bar,1]
        """
        pattern = feature['midi_pattern']
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[0] * resolution * 4 / 2**(time_sig[1])
                if num_bar is None:
                    num_bar = int(round(float(pattern[track_num][-1].tick) / bar_length))
                    used_notes = np.zeros((num_bar, 1))
                else:
                    used_notes = np.zeros((num_bar, 1))

            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                if 'time_sig' not in locals():  # set default bar length as 4 beat
                    bar_length = 4 * resolution
                    time_sig = [4, 2, 24, 8]

                    if num_bar is None:
                        num_bar = int(round(float(pattern[track_num][-1].tick) / bar_length))
                        used_notes = np.zeros((num_bar, 1))
                        used_notes[pattern[track_num][i].tick / bar_length] += 1
                    else:
                        used_notes = np.zeros((num_bar, 1))
                        used_notes[pattern[track_num][i].tick / bar_length] += 1
                    note_list = []
                    note_list.append(pattern[track_num][i].data[0])

                else:
                    for j in range(0, num_bar):
                        if 'note_list'in locals():
                            pass
                        else:
                            note_list = []
                    note_list.append(pattern[track_num][i].data[0])
                    used_notes[pattern[track_num][i].tick / bar_length] += 1

        used_pitch = np.zeros((num_bar, 1))
        current_note = 0
        for i in range(0, num_bar):
            used_pitch[i] = len(set(note_list[current_note:current_note + int(used_notes[i][0])]))
            current_note += int(used_notes[i][0])

        return used_pitch

    def total_used_note(self, feature, track_num=1):
        """
        total_used_note (Note count): The number of used notes.
        As opposed to the pitch count, the note count does not contain pitch information but is a rhythm-related feature.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        'used_notes': a scalar for each sample.
        """
        pattern = feature['midi_pattern']
        used_notes = 0
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                used_notes += 1
        return used_notes

    def bar_used_note(self, feature, track_num=1, num_bar=None):
        """
        bar_used_note (Note count per bar).

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.

        Returns:
        'used_notes': with shape of [num_bar, 1]
        """
        pattern = feature['midi_pattern']
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[track_num] * resolution * 4 / 2**(time_sig[1])
                if num_bar is None:
                    num_bar = int(round(float(pattern[track_num][-1].tick) / bar_length))
                    used_notes = np.zeros((num_bar, 1))
                else:
                    used_notes = np.zeros((num_bar, 1))

            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                if 'time_sig' not in locals():  # set default bar length as 4 beat
                    bar_length = 4 * resolution
                    time_sig = [4, 2, 24, 8]

                    if num_bar is None:
                        num_bar = int(round(float(pattern[track_num][-1].tick) / bar_length))
                        used_notes = np.zeros((num_bar, 1))
                        used_notes[pattern[track_num][i].tick / bar_length] += 1
                    else:
                        used_notes = np.zeros((num_bar, 1))
                        used_notes[pattern[track_num][i].tick / bar_length] += 1

                else:
                    used_notes[pattern[track_num][i].tick / bar_length] += 1
        return used_notes

    def total_pitch_class_histogram(self, feature):
        """
        total_pitch_class_histogram (Pitch class histogram):
        The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
        In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

        Returns:
        'histogram': histrogram of 12 pitch, with weighted duration shape 12
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        histogram = np.zeros(12)
        for i in range(0, 128):
            pitch_class = i % 12
            histogram[pitch_class] += np.sum(piano_roll, axis=1)[i]
        histogram = histogram / sum(histogram)
        return histogram

    def bar_pitch_class_histogram(self, feature, track_num=1, bpm=120, num_bar=None):
        """
        bar_pitch_class_histogram (Pitch class histogram per bar):

        Args:
        'bpm' : specify the assigned speed in bpm, default is 120 bpm.
        'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        'histogram': with shape of [num_bar, 12]
        """

        # todo: deal with more than one time signature cases
        pm_object = feature['pretty_midi']
        if num_bar is None:
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_length = 60. / bpm * numer * 4 / deno * 100
            piano_roll = pm_object.instruments[track_num].get_piano_roll(fs=100)
            piano_roll = np.transpose(piano_roll, (1, 0))
            actual_bar = len(piano_roll) / bar_length
            num_bar = int(round(actual_bar))
            bar_length = int(round(bar_length))
        else:
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_length = 60. / bpm * numer * 4 / deno * 100
            piano_roll = pm_object.instruments[track_num].get_piano_roll(fs=100)
            piano_roll = np.transpose(piano_roll, (1, 0))
            actual_bar = len(piano_roll) / bar_length
            bar_length = int(math.ceil(bar_length))

        if actual_bar > num_bar:
            piano_roll = piano_roll[:-np.mod(len(piano_roll), bar_length)].reshape((num_bar, -1, 128))  # make exact bar
        elif actual_bar == num_bar:
            piano_roll = piano_roll.reshape((num_bar, -1, 128))
        else:
            piano_roll = np.pad(piano_roll, ((0, int(num_bar * bar_length - len(piano_roll))), (0, 0)), mode='constant', constant_values=0)
            piano_roll = piano_roll.reshape((num_bar, -1, 128))

        bar_histogram = np.zeros((num_bar, 12))
        for i in range(0, num_bar):
            histogram = np.zeros(12)
            for j in range(0, 128):
                pitch_class = j % 12
                histogram[pitch_class] += np.sum(piano_roll[i], axis=0)[j]
            if sum(histogram) != 0:
                bar_histogram[i] = histogram / sum(histogram)
            else:
                bar_histogram[i] = np.zeros(12)
        return bar_histogram

    def pitch_class_transition_matrix(self, feature, normalize=0):
        """
        pitch_class_transition_matrix (Pitch class transition matrix):
        The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition.
        The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes.

        Args:
        'normalize' : If set to 0, return transition without normalization.
                      If set to 1, normalizae by row.
                      If set to 2, normalize by entire matrix sum.
        Returns:
        'transition_matrix': shape of [12, 12], transition_matrix of 12 x 12.
        """
        pm_object = feature['pretty_midi']
        transition_matrix = pm_object.get_pitch_class_transition_matrix()

        if normalize == 0:
            return transition_matrix

        elif normalize == 1:
            sums = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1, 1)

        elif normalize == 2:
            return transition_matrix / sum(sum(transition_matrix))

        else:
            print "invalid normalization mode, return unnormalized matrix"
            return transition_matrix

    def pitch_range(self, feature):
        """
        pitch_range (Pitch range):
        The pitch range is calculated by subtraction of the highest and lowest used pitch in semitones.

        Returns:
        'p_range': a scalar for each sample.
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        pitch_index = np.where(np.sum(piano_roll, axis=1) > 0)
        p_range = np.max(pitch_index) - np.min(pitch_index)
        return p_range

    def avg_pitch_shift(self, feature, track_num=1):
        """
        avg_pitch_shift (Average pitch interval):
        Average value of the interval between two consecutive pitches in semitones.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        'pitch_shift': a scalar for each sample.
        """
        pattern = feature['midi_pattern']
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        total_used_note = self.total_used_note(pattern, track_num=track_num)
        d_note = np.zeros((total_used_note - 1))
        current_note = 0
        counter = 0
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                if counter != 0:
                    d_note[counter - 1] = current_note - pattern[track_num][i].data[0]
                    current_note = pattern[track_num][i].data[0]
                    counter += 1
                else:
                    current_note = pattern[track_num][i].data[0]
                    counter += 1
        pitch_shift = np.mean(abs(d_note))
        return pitch_shift

    def avg_IOI(self, feature):
        """
        avg_IOI (Average inter-onset-interval):
        To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.

        Returns:
        'avg_ioi': a scalar for each sample.
        """

        pm_object = feature['pretty_midi']
        onset = pm_object.get_onsets()
        ioi = np.diff(onset)
        avg_ioi = np.mean(ioi)
        return avg_ioi

    def note_length_hist(self, feature, track_num=1, normalize=True, pause_event=False):
        """
        note_length_hist (Note length histogram):
        To extract the note length histogram, we first define a set of allowable beat length classes:
        [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet].
        The pause_event option, when activated, will double the vector size to represent the same lengths for rests.
        The classification of each event is performed by dividing the basic unit into the length of (barlength)/96, and each note length is quantized to the closest length category.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'normalize' : If true, normalize by vector sum.
        'pause_event' : when activated, will double the vector size to represent the same lengths for rests.

        Returns:
        'note_length_hist': The output vector has a length of either 12 (or 24 when pause_event is True).
        """

        pattern = feature['midi_pattern']
        if pause_event is False:
            note_length_hist = np.zeros((12))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            # basic unit: bar_length/96
            for i in range(0, len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num] * resolution * 4 / 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                    if 'time_sig' not in locals():  # set default bar length as 4 beat
                        bar_length = 4 * resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length / 96.
                    hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    # find next note off
                    for j in range(i, len(pattern[track_num])):
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - current_tick
                                distance = np.abs(np.array(hist_list) - note_length)
                                idx = distance.argmin()
                                note_length_hist[idx] += 1
                                break
        else:
            note_length_hist = np.zeros((24))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            # basic unit: bar_length/96
            for i in range(0, len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num] * resolution * 4 / 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                    check_previous_off = True
                    if 'time_sig' not in locals():  # set default bar length as 4 beat
                        bar_length = 4 * resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length / 96.
                    tol = 3. * unit
                    hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    # find next note off
                    for j in range(i, len(pattern[track_num])):
                        # find next note off
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - current_tick
                                distance = np.abs(np.array(hist_list) - note_length)
                                idx = distance.argmin()
                                note_length_hist[idx] += 1
                                break
                            else:
                                if pattern[track_num][j].tick == current_tick:
                                    check_previous_off = False

                    # find previous note off/on
                    if check_previous_off is True:
                        for j in range(i - 1, 0, -1):
                            if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] != 0:
                                break

                            elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):

                                note_length = current_tick - pattern[track_num][j].tick
                                distance = np.abs(np.array(hist_list) - note_length)
                                idx = distance.argmin()
                                if distance[idx] < tol:
                                    note_length_hist[idx + 12] += 1
                                break

        if normalize is False:
            return note_length_hist

        elif normalize is True:

            return note_length_hist / np.sum(note_length_hist)

    def note_length_transition_matrix(self, feature, track_num=1, normalize=0, pause_event=False):
        """
        note_length_transition_matrix (Note length transition matrix):
        Similar to the pitch class transition matrix, the note length tran- sition matrix provides useful information for rhythm description.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'normalize' : If true, normalize by vector sum.
        'pause_event' : when activated, will double the vector size to represent the same lengths for rests.

        'normalize' : If set to 0, return transition without normalization.
                      If set to 1, normalizae by row.
                      If set to 2, normalize by entire matrix sum.

        Returns:
        'transition_matrix': The output feature dimension is 12 × 12 (or 24 x 24 when pause_event is True).
        """
        pattern = feature['midi_pattern']
        if pause_event is False:
            transition_matrix = np.zeros((12, 12))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            idx = None
            # basic unit: bar_length/96
            for i in range(0, len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num] * resolution * 4 / 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                    if 'time_sig' not in locals():  # set default bar length as 4 beat
                        bar_length = 4 * resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length / 96.
                    hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    # find note off
                    for j in range(i, len(pattern[track_num])):
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                            if pattern[track_num][j].data[0] == current_note:
                                note_length = pattern[track_num][j].tick - current_tick
                                distance = np.abs(np.array(hist_list) - note_length)

                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx is not None:
                                    transition_matrix[last_idx][idx] += 1
                                break
        else:
            transition_matrix = np.zeros((24, 24))
            pattern.make_ticks_abs()
            resolution = pattern.resolution
            idx = None
            # basic unit: bar_length/96
            for i in range(0, len(pattern[track_num])):
                if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                    time_sig = pattern[track_num][i].data
                    bar_length = time_sig[track_num] * resolution * 4 / 2**(time_sig[1])
                elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                    check_previous_off = True
                    if 'time_sig' not in locals():  # set default bar length as 4 beat
                        bar_length = 4 * resolution
                        time_sig = [4, 2, 24, 8]
                    unit = bar_length / 96.
                    tol = 3. * unit
                    hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit * 72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                    current_tick = pattern[track_num][i].tick
                    current_note = pattern[track_num][i].data[0]
                    # find next note off
                    for j in range(i, len(pattern[track_num])):
                        # find next note off
                        if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                            if pattern[track_num][j].data[0] == current_note:

                                note_length = pattern[track_num][j].tick - current_tick
                                distance = np.abs(np.array(hist_list) - note_length)
                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx is not None:
                                    transition_matrix[last_idx][idx] += 1
                                break
                            else:
                                if pattern[track_num][j].tick == current_tick:
                                    check_previous_off = False

                    # find previous note off/on
                    if check_previous_off is True:
                        for j in range(i - 1, 0, -1):
                            if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] != 0:
                                break

                            elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):

                                note_length = current_tick - pattern[track_num][j].tick
                                distance = np.abs(np.array(hist_list) - note_length)

                                last_idx = idx
                                idx = distance.argmin()
                                if last_idx is not None:
                                    if distance[idx] < tol:
                                        idx = last_idx
                                        transition_matrix[last_idx][idx + 12] += 1
                                break

        if normalize == 0:
            return transition_matrix

        elif normalize == 1:

            sums = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1, 1)

        elif normalize == 2:

            return transition_matrix / sum(sum(transition_matrix))

        else:
            print "invalid normalization mode, return unnormalized matrix"
            return transition_matrix

    def chord_dependency(self, feature, bar_chord, bpm=120, num_bar=None, track_num=1):
        pm_object = feature['pretty_midi']
        # compare bar chroma with chord chroma. calculate the ecludian
        bar_pitch_class_histogram = self.bar_pitch_class_histogram(pm_object, bpm=bpm, num_bar=num_bar, track_num=track_num)
        dist = np.zeros((len(bar_pitch_class_histogram)))
        for i in range((len(bar_pitch_class_histogram))):
            dist[i] = np.linalg.norm(bar_pitch_class_histogram[i] - bar_chord[i])
        average_dist = np.mean(dist)
        return average_dist
