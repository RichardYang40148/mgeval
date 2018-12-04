# mgeval

An Objective evaluation toolbox for symbolic domain music generation.

The work is published as [On the evaluation of generative models in music](https://link.springer.com/article/10.1007/s00521-018-3849-7?wt_mc=Internal.Event.1.SEM.ArticleAuthorOnlineFirst&utm_source=ArticleAuthorOnlineFirst&utm_medium=email&utm_content=AA_en_06082018&ArticleAuthorOnlineFirst_20181106)

The article is provided as [journal online access](https://link.springer.com/epdf/10.1007/s00521-018-3849-7?author_access_token=Z7YxQv2K9z33nk1_XGlY9_e4RwlQNchNByi7wbcMAY5M_T6iwDlmVavmHfG20IIuk492IRWVj17BK1zhOxg5HA5fo8df4mI0b3U1YbTvprNarTF7BunHbKBquKplW2anwIy_TzUtUKq8g6tZzhCUzQ%3D%3D) and [downloadable preprint](http://www.musicinformatics.gatech.edu/wp-content_nondefault/uploads/2018/11/postprint.pdf)

## Introduction:

### Why?
Machine learning and neural networks have demonsrate great potential in music generation. However, vary goals and definition of creativity makes evaluation toward generative music challenging. In recent research, subjective evaluations seem to the only viable way to do this.

The goal of this toolbox is to provide an objective evaluation system to have a better reliability, validity, and replicability in the music generation.

### How?

The toolbox propose a set of musically informed objective measures to obtain analyses that further driven series of evaluation strategy that involves:

1. Absolute metrics that can provide insights into properties of generated or collected set of data
 
2. Relative metrics in order to compare two sets of data, e.g., training and generated.

### Musically informed objective measures

1. Pitch count: The number of different pitches within a sample. The output is a scalar for each sample.
2. Pitch class histogram: The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale. In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.
3. Pitch class transition matrix: The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition. The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes. The resulting feature dimensionality is 12 × 12.
4. Pitch range: The pitch range is calculated by subtraction of the highest and lowest used pitch in semitones. The output is a scalar for each sample.
5. Average pitch interval: Average value of the interval between two consecutive pitches in semitones. The output is a scalar for each sample.

1. Note count: The number of used notes. As opposed to the pitch count, the note count does not contain pitch information but is a rhythm-related feature. The output is a scalar for each sample.
2. Average inter-onset-interval: To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes. The output is a scalar in seconds for each sample.
3. Note length histogram: To extract the note length histogram, we first define a set of allowable beat length classes [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet]. The rest option, when activated, will double the vector size to represent the same lengths for rests. The classification of each event is performed by dividing the basic unit into the length of (barlength)/96, and each note length is quantized to the closest length category. The output vector has a length of either 12 or 24, respectively.
4. Note length transition matrix: Similar to the pitch class transition matrix, the note length tran- sition matrix provides useful information for rhythm description. The output feature dimension is 12 × 12 or 24 × 24, respectively.

### Dependency
 * [`scipy`](http://www.scipy.org/)
 * [`numpy`](http://www.numpy.org)
 * [`pretty-midi`](https://github.com/craffel/pretty-midi)
 * [`python-midi`](https://github.com/vishnubob/python-midi)
 * [`sklearn`](https://scikit-learn.org/stable/)
 ### To use
 Detailed documentation included [here](https://github.com/RichardYang40148/mgeval/blob/master/mgeval/documentation.md).
 
 Demonstration of metrics included in [demo.ipynb](https://github.com/RichardYang40148/mgeval/commits?author=RichardYang40148)
