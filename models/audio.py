import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import lws

class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


# Default hyperparameters
hp = HParams(
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality # 80 个梅尔频度
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,

    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter #  FFT点数
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate) # 窗体移动距离，每移动一下hop_size，窗口包含的范围作为一个新的帧进行 fft
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)  # 窗口长度
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    # Icey sample_rate/hop_size=本次采样16khz分成的总帧数

    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
    # be too big to avoid gradient explosion,
    # not too small for fast convergence)
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.

    ###################### Our training parameters #################################
    img_size=288,
    fps=25,

    batch_size=8,
    initial_learning_rate=1e-4,
    nepochs=200000000000000000,
    ### ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
    num_workers=4,
    checkpoint_interval=6000,
    eval_interval=6000,
    save_optimizer_state=True,

    syncnet_wt=0.0,  # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence.
    syncnet_batch_size=128,
    syncnet_lr=1e-4,
    syncnet_eval_interval=4500,
    syncnet_checkpoint_interval=4500,

    disc_wt=0.07,
    disc_initial_learning_rate=1e-4,
)


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav) # Icey 一阶无限冲击响应(IIR)滤波器
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize)) # 使用短时傅里叶变换得到频谱
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")

# Icey  音频信号 y 的短时傅里叶变换 (STFT)
def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2 # 用断言确保 hp.fmax 小于等于采样率的一半，以避免超出 Nyquist 频率范围
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)
# hp.sample_rate: 采样率
# hp.n_fft: FFT 窗口的大小
# n_mels=hp.num_mels: 梅尔滤波器的数量
# fmin=hp.fmin: 最低频率
# fmax=hp.fmax: 最高频率

# Icey 将振幅值转换为分贝(dB)刻度
def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
