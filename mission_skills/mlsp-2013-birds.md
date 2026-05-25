# Blueprint: MLSP 2013 — Bird Song Recognition (Audio Feature Extraction & Augmentation)

> ⚠️ **Nature of this notebook pair: Audio EDA / Feature Extraction + Augmentation only.**
> There is no model training, no training loop, and no submission file generated.
> This blueprint combines two companion notebooks:
> - `mlsp-2013-birds.ipynb` — Audio feature extraction using librosa (time-domain, frequency-domain, spectrum-based)
> - `audio-albumentations-torchaudio-audiomentations.ipynb` — Audio augmentation using torchaudio and audiomentations
>
> **Note on data source:** Both notebooks load data from the **BirdCLEF 2021** competition
> (`birdclef-2021/train_metadata.csv`). The techniques are applicable to MLSP 2013 and any bird audio competition.

---

## Competition Details

| Field | Value |
|---|---|
| Filename references | [MLSP 2013 Bird Song Recognition](https://www.kaggle.com/competitions/mlsp-2013-birds) |
| Actual data used | [BirdCLEF 2021](https://www.kaggle.com/competitions/birdclef-2021) |
| Task | Multi-label bird species identification from audio recordings |
| Input | OGG/WAV audio files of bird calls |
| Evaluation | (Not addressed in this notebook) |

---

## Dependencies

| Library | Notebook | Role |
|---|---|---|
| `librosa` | Feature extraction | Core audio loading, feature extraction, visualization |
| `torchaudio` | Both | Audio I/O, transforms (Resample, Spectrogram, TimeMasking, FrequencyMasking, Fade, Vol, GriffinLim) |
| `torchaudio.sox_effects` | Augmentation | Speed change, reverberation, channel remix |
| `audiomentations` | Augmentation | TimeStretch, ClippingDistortion, PitchShift, PolarityInversion, Shift |
| `torch` | Both | Tensor operations |
| `IPython.display` | Both | In-notebook audio playback |
| `librosa.display` | Feature extraction | Spectrogram and waveform visualization |
| `matplotlib`, `seaborn`, `plotly` | Both | Visualization |
| `sklearn.preprocessing` | Feature extraction | `minmax_scale` for normalizing feature plots |
| `numpy`, `pandas` | Both | Data handling |
| `geopandas`, `shapely` | Augmentation | Imported but not used substantively |

Install: `!pip install audiomentations`

---

## Data

- **Source:** `birdclef-2021/train_metadata.csv` and `train_short_audio/`
- **Sample species analyzed:** `astfly`, `casvir`, `subfly`, `wilfly`, `verdin`, `solsan` (6 birds)
- **Loading (librosa):** `librosa.load()` — returns float32 NumPy waveform `y` and native sample rate `sr` (default 22,050 Hz)
- **Loading (torchaudio):** `torchaudio.load()` — returns float32 PyTorch Tensor of shape `(channels, frames)` and sample rate
- **Silence trimming:** `librosa.effects.trim(y)` — removes leading/trailing silence before feature extraction

---

## Audio Features Covered

### 1. Time Domain Features

#### Waveform Visualization
```python
librosa.display.waveplot(y=audio, sr=sr)
```
Plots amplitude vs. time for all 6 species on the same figure.

#### Log-Frequency Spectrogram (STFT)
```python
n_fft = 2048
hop_length = 512
D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
```
STFT segments the signal into overlapping windows and computes per-window Fourier transforms. The log-frequency axis makes harmonic relationships visible.

#### RMSE (Root Mean Square Energy)
```python
S, phase = librosa.magphase(librosa.stft(audio))
rms = librosa.feature.rms(S=S)
```
Characterizes signal loudness per frame. Plotted alongside the log power spectrogram to show energy variation over time.

#### Mel Spectrogram
```python
S = librosa.feature.melspectrogram(audio, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)
```
Converts frequency axis to the perceptual mel scale, which compresses high frequencies to match human pitch perception.

#### Zero Crossing Rate (ZCR)
```python
zero_crossings = librosa.zero_crossings(audio, pad=False)
zcr_count = sum(zero_crossings)
```
Counts how often the waveform crosses zero amplitude. High ZCR indicates unvoiced/noisy segments; low ZCR indicates voiced/tonal content. Compared across all 6 species.

#### Harmonic-Percussive Source Separation (HPSS)
```python
y_harm, y_perc = librosa.effects.hpss(audio)
H, P = librosa.decompose.hpss(librosa.stft(audio))
```
Decomposes signal into harmonic (pitched, horizontal spectrogram patterns) and percussive (transient, vertical patterns) components. All three spectrograms (full, harmonic, percussive) plotted for `casvir`.

#### Beat Extraction / BPM
```python
tempo, beat_frames = librosa.beat.beat_track(y=y_harm, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
beat_time_diff = np.ediff1d(beat_times)
```
BPM extracted from the harmonic component of each species; inter-beat intervals plotted as bar charts.

---

### 2. Frequency Domain Features

#### Chromagram (chroma_stft)
```python
chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
```
12-bin representation of spectral energy per pitch class (C, C#, D, …, B). Captures tonal/harmonic structure; useful for melody and chord similarity.

#### Constant Q-Transform (CQT)
```python
chroma_cq = librosa.feature.chroma_cqt(y=audio, sr=sr)
```
Similar to STFT but with logarithmically spaced frequency bins — better suited to music where notes span multiple octaves. Compared against `chroma_stft` for `casvir`.

#### Chroma Energy Normalized Statistics (CENS)
```python
chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
```
Smoothed and normalized version of chroma features. More robust to dynamics and articulation differences; typically used for audio matching and similarity retrieval.

---

### 3. Spectrum-Based Features

#### Spectral Centroid
```python
spectral_centroids = librosa.feature.spectral_centroid(audio, sr=sr)[0]
```
Weighted mean frequency of the spectrum — the "center of mass." High centroid = brighter sound. Overlaid on the waveform after min-max normalization.

#### Spectral Contrast
```python
contrast = librosa.feature.spectral_contrast(y=y_harm, sr=sr)
```
Measures the difference between spectral peaks (harmonic partials) and valleys (noise) across sub-bands. 7 sub-bands by default; high contrast = more tonal content.

#### Spectral Rolloff
```python
spectral_rolloff = librosa.feature.spectral_rolloff(audio, sr=sr)[0]
```
Frequency below which 85% of the total spectral energy is concentrated. Distinguishes harmonic (low rolloff) from noisy (high rolloff) sounds. Overlaid on the waveform.

#### MFCCs (Mel-Frequency Cepstral Coefficients)
```python
mfcc = librosa.feature.mfcc(y=audio, sr=sr)  # shape: (20, T)
```
20 coefficients capturing the overall spectral envelope shape. Most widely used audio feature in speech and bird call recognition. Displayed as a heatmap over time.

---

## Audio Augmentations Covered

> Source: `audio-albumentations-torchaudio-audiomentations.ipynb`
> All augmentations are demonstrated with before/after waveform plots and in-notebook audio playback.

### torchaudio Transforms

#### Resample
```python
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)
# new_sample_rate = sample_rate / 10  (downsample by 10x)
```
Changes the number of samples per second. Used for speed-invariant feature testing.

#### Speed Change + Reverberation (SoX Effects)
```python
effects = [
    ["speed", "1.2"],          # increase playback speed (changes pitch)
    ["rate", f"{sample_rate}"],# restore original sample rate after speed change
    ["reverb", "-w"],          # add wet reverberation
]
waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
    waveform, sample_rate, effects
)
```
SoX effect chains apply multiple transforms in sequence. Speed + rate combination shifts pitch without stretching duration. `-w` enables the "wet" reverb mode.

#### Background Noise Addition (SNR-based)
```python
speech_power = speech.norm(p=2)
noise_power  = noise.norm(p=2)
snr_db = 20
snr = math.exp(snr_db / 10)
scale = snr * noise_power / speech_power
noisy_speech = (scale * speech + noise) / 2
```
Mixes a bird call with noise at a controlled Signal-to-Noise Ratio (SNR = 20 dB). Audio is resampled to 6,000 Hz before mixing. SNR controls how loud the signal is relative to the noise.

#### SpecAugment — Time Masking
```python
n_fft, hop_length = 2048, 400
spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)(waveform)
spec = T.TimeMasking(time_mask_param=1300)(spec)
# Reconstruct waveform from masked spectrogram:
waveform_masked = T.GriffinLim(n_fft=n_fft, hop_length=hop_length)(spec)
```
Zeros out a contiguous block of up to 1,300 time frames. GriffinLim reconstructs audio from the masked spectrogram for listening verification.

#### SpecAugment — Frequency Masking
```python
spec = T.FrequencyMasking(freq_mask_param=1000)(spec)
waveform_masked = T.GriffinLim(n_fft=n_fft, hop_length=hop_length)(spec)
```
Zeros out a contiguous block of up to 1,000 frequency bins. Together with time masking, replicates the full SpecAugment technique.

#### Fade In / Fade Out
```python
fade = T.Fade(fade_in_len=200, fade_out_len=100, fade_shape='linear')
waveform_faded = fade(waveform)
```
Applies a linear amplitude ramp at the start (200 samples) and end (100 samples) of the waveform. Useful for preventing click artifacts at clip boundaries.

#### Volume Transform
```python
vol = T.Vol(gain=29, gain_type='db')
waveform_loud = vol(waveform)
```
Scales the waveform amplitude by a fixed gain in decibels. Simulates recordings made at different distances or microphone sensitivities.

---

### audiomentations Transforms

All audiomentations transforms operate on NumPy arrays and are composed via `Compose([...])`.

#### Time Stretch + Clipping Distortion
```python
from audiomentations import TimeStretch, ClippingDistortion, Compose

augmenter = Compose([
    ClippingDistortion(min_percentile_threshold=20, max_percentile_threshold=40, p=1.0),
    TimeStretch(min_rate=0.8, max_rate=0.9, leave_length_unchanged=False, p=1.0),
])
waveform_aug = augmenter(samples=waveform.numpy(), sample_rate=sample_rate)
```
- **ClippingDistortion:** Hard-clips amplitude values above the 20th–40th percentile threshold, simulating overdriven microphones or noisy field recorders
- **TimeStretch:** Stretches the audio to 80–90% of original speed without changing pitch (`leave_length_unchanged=False` means output duration changes)

#### Pitch Shift + Polarity Inversion
```python
from audiomentations import PitchShift, PolarityInversion

augmenter = Compose([
    PitchShift(min_semitones=-2, max_semitones=-1, p=1.0),
    PolarityInversion(p=1.0),
])
```
- **PitchShift:** Shifts pitch down by 1–2 semitones without changing tempo; simulates seasonal variation in bird call pitch
- **PolarityInversion:** Flips waveform sign (`y → -y`); inaudible to humans but changes phase relationships, acting as a free regularization augmentation

#### Time Shift (Forward & Backward)
```python
from audiomentations import Shift

# Forward shift: move signal 50% forward in time
forward_augmenter = Compose([Shift(min_fraction=0.5, max_fraction=0.5, p=1.0)])

# Backward shift: move signal 25% backward in time
backward_augmenter = Compose([Shift(min_fraction=-0.25, max_fraction=-0.25, p=1.0)])
```
Cyclically shifts the waveform left or right in time. Positive fraction shifts forward (content wraps from end to beginning); negative fraction shifts backward. Simulates different recording start offsets within a longer clip.

---

## Augmentation Summary Table

| Augmentation | Library | Applied to | Effect |
|---|---|---|---|
| Resample | torchaudio | Waveform | Changes sample rate (speed/quality) |
| Speed + Reverb | torchaudio (SoX) | Waveform | Pitch shift + room simulation |
| Background Noise | torchaudio | Waveform | SNR-controlled noise injection |
| Time Masking | torchaudio | Spectrogram | Zeros contiguous time bands |
| Frequency Masking | torchaudio | Spectrogram | Zeros contiguous frequency bands |
| Fade In/Out | torchaudio | Waveform | Amplitude ramp at clip edges |
| Volume | torchaudio | Waveform | Gain adjustment (dB) |
| Time Stretch | audiomentations | Waveform | Tempo change, duration varies |
| Clipping Distortion | audiomentations | Waveform | Hard amplitude clipping |
| Pitch Shift | audiomentations | Waveform | Pitch shift (semitones), fixed tempo |
| Polarity Inversion | audiomentations | Waveform | Sign flip, free regularization |
| Forward/Backward Shift | audiomentations | Waveform | Cyclic time offset |

---

## Output

This notebook pair produces no saved model files and no submission CSV. One exception: the augmentation notebook saves the original `astfly` audio as `./audio.mp3` via `torchaudio.save()` (format conversion demo only). All outputs are in-notebook visualizations and printed statistics. No submission CSV is generated.

---

## Key Observations

| Feature | What it reveals for bird calls |
|---|---|
| Mel spectrogram | Frequency band usage; bird species tend to sing in characteristic frequency ranges |
| MFCC | Compact representation of call timbre; ~20 coefficients encode most discriminating info |
| Chromagram | Tonal structure; some bird songs have strong pitch periodicity |
| ZCR | Distinguishes tonal song segments from noisy ambient sounds |
| HPSS | Isolates clean harmonic song from percussive background noise |
| Spectral centroid | Bright vs. dark call character; small birds tend to call at higher centroids |

---

## What a Real Baseline Needs

A competitive bird classification pipeline would require:

1. **Segmentation:** Split long recordings into fixed-length clips (e.g., 5-second windows); discard silence via VAD or ZCR thresholding (ZCR covered in the feature extraction notebook)
2. **Feature representation:** Compute mel spectrograms (128 mel bins, hop 512, n_fft 2048) as model input — mel spectrogram and MFCC extraction are both covered in the feature extraction notebook
3. **Data augmentation pipeline:** Combine augmentations from the augmentation notebook:
   - Waveform-level: pitch shift, time stretch, background noise (SNR-based), polarity inversion, time shift
   - Spectrogram-level: time masking, frequency masking (SpecAugment), volume scaling
   - Apply HPSS first to isolate the harmonic bird call from percussive background before augmenting
4. **Model:** CNN (EfficientNet/ResNet) on mel spectrogram images, or 1D CNN / Wav2Vec2 on raw audio
5. **Multi-label handling:** Each clip may contain multiple species; use BCEWithLogitsLoss with sigmoid output
6. **Evaluation:** Row-wise macro-averaged ROC-AUC (BirdCLEF) or average precision (MLSP 2013)
7. **Submission:** Per-clip probability scores for each species class
