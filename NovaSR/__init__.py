import torch
import os
import librosa
from .speechsr import SynthesizerTrn
from torch.nn.utils import weight_norm

class FastSR:
    def __init__(self, ckpt_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hps = {
            "train": {
                "segment_size": 9600
            },
            "data": {
                "hop_length": 320,
                "n_mel_channels": 128
            },
            "model": {
                "resblock": "0",
                "resblock_kernel_sizes": [11],
                "resblock_dilation_sizes": [[1,3,5]],
                "upsample_initial_channel": 32,
            }
        }
        if ckpt_path is None:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download("YatharthS/NovaSR")
            ckpt_path = f"{model_path}/pytorch_model_v1.bin"

        self.model = self._load_model(ckpt_path).half().eval()


    def _load_model(self, ckpt_path):
        model = SynthesizerTrn(
            self.hps['data']['n_mel_channels'],
            self.hps['train']['segment_size'] // self.hps['data']['hop_length'],
            **self.hps['model']
        ).to(self.device)
        assert os.path.isfile(ckpt_path)
        checkpoint_dict = torch.load(ckpt_path, map_location='cpu')
        model.dec.remove_weight_norm()
        model.load_state_dict(checkpoint_dict, strict=True)
        model.eval()
        return model

    def load_audio(self, audio_file):
        y, sr = librosa.load(audio_file, sr=16000) ## resample to 16khz sr
        lowres_wav = torch.from_numpy(y).unsqueeze(0).half().unsqueeze(1).to(self.device)
        return lowres_wav
        
    def infer(self, lowres_wav):
        with torch.no_grad():
            new_wav = self.model(lowres_wav)

        return new_wav.squeeze(0)

        
