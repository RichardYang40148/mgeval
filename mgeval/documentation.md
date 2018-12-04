# core.py

Include feature extractor and musically informed objective measures.

## extract_feature:
This function extracts two midi feature:
[pretty_midi object](https://github.com/craffel/pretty-midi) and 
[midi_pattern](https://github.com/vishnubob/python-midi)

```
Returns:
    dict(pretty_midi: pretty_midi object,
         midi_pattern: midi pattern contains a list of tracks)
```

## metrics (Include feature extractor and musically informed objective measures)

### total_used_pitch (Pitch count):
The number of different pitches within a sample.

```
Returns:
'used_pitch': pitch count, scalar for each sample.
```

### bar_used_pitch (Pitch count per bar)

```
Args:
'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.

Returns:
'used_pitch': with shape of [num_bar,1]
```

### total_used_note (Note count): 

The number of used notes.
As opposed to the pitch count, the note count does not contain pitch information but is a rhythm-related feature.
```
Args:
'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

Returns:
'used_notes': a scalar for each sample.
```


### bar_used_note (Note count per bar).

```
Args:
'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.

Returns:
'used_notes': with shape of [num_bar, 1]
```
        
### total_pitch_class_histogram (Pitch class histogram):
The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.
```
Returns:
'histogram': histrogram of 12 pitch, with weighted duration shape 12
```

### bar_pitch_class_histogram (Pitch class histogram per bar):

```
Args:
'bpm' : specify the assigned speed in bpm, default is 120 bpm.
'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.
'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

Returns:
'histogram': with shape of [num_bar, 12]
```

### pitch_class_transition_matrix (Pitch class transition matrix):
The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition.
The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes.
```
Args:
'normalize' : If set to 0, return transition without normalization.
              If set to 1, normalizae by row.
              If set to 2, normalize by entire matrix sum.
Returns:
'transition_matrix': shape of [12, 12], transition_matrix of 12 x 12.
```

### pitch_range (Pitch range):

The pitch range is calculated by subtraction of the highest and lowest used pitch in semitones.
```
Returns:
'p_range': a scalar for each sample.
```

### avg_pitch_shift (Average pitch interval):

Average value of the interval between two consecutive pitches in semitones.
```
Args:
'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

Returns:
'pitch_shift': a scalar for each sample.
```

### avg_IOI (Average inter-onset-interval):
To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.
```
Returns:
'avg_ioi': a scalar for each sample.
```

### note_length_hist (Note length histogram):

To extract the note length histogram, we first define a set of allowable beat length classes:
[full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet].
The pause_event option, when activated, will double the vector size to represent the same lengths for rests.
The classification of each event is performed by dividing the basic unit into the length of (barlength)/96, and each note length is quantized to the closest length category.
```
Args:
'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
'normalize' : If true, normalize by vector sum.
'pause_event' : when activated, will double the vector size to represent the same lengths for rests.

Returns:
'note_length_hist': The output vector has a length of either 12 (or 24 when pause_event is True).
```


### note_length_transition_matrix (Note length transition matrix):

Similar to the pitch class transition matrix, the note length tran- sition matrix provides useful information for rhythm description.
```
Args:
'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
'normalize' : If true, normalize by vector sum.
'pause_event' : when activated, will double the vector size to represent the same lengths for rests.

'normalize' : If set to 0, return transition without normalization.
              If set to 1, normalizae by row.
              If set to 2, normalize by entire matrix sum.

Returns:
'transition_matrix': The output feature dimension is 12 × 12 (or 24 x 24 when pause_event is True).
```
