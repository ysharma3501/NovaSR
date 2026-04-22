## NovaSR: Pushing the Limits of Extreme Efficiency in Audio Super-Resolution

> [!TIP]
> Please check out [LavaSR](https://github.com/ysharma3501/LavaSR.git), it has several major benefits over NovaSR
>
> - NovaSR: only 16khz input, ~52KB, ~3000× realtime speed, low-medium quality 
> - LavaSR: supports **any input from 8–48kHz**, **~5000× realtime speed**, 50MB, and significantly better quality(surpasses 6gb models)
>
> If you were using the old fast model, try switching to the new one, you’ll get much higher fidelity, flexible input rates, and faster speeds.
> 
<p align="center">
  <a href="https://huggingface.co/YatharthS/NovaSR">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/YatharthS/NovaSR">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
  &nbsp;
  <a href="https://www.kaggle.com/code/yatharthsharma888/novasr-training">
    <img src="https://img.shields.io/badge/Kaggle-Training%20Notebook-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle Notebook">
  </a>
</p>


This is the repository for NovaSR, a tiny 50kB audio upsampling model that upscales muffled 16khz audio into clear and crisp 48khz audio at speeds over 3500x realtime. 


https://github.com/user-attachments/assets/c81f87eb-f6de-4bf9-85bd-dfc9a223a865


### Key benefits
* Speed: Can reach 3600x realtime speed on a single a100 gpu.
* Quality: On par with models 5,000x larger.
* Size: Just 52kB in size, several thousand times smaller then most. 

### Why is this even useful?
* Enhancing models: NovaSR can enhance TTS model quality considerably with nearly 0 computational cost.
* Real-time enhancement: NovaSR allows for on device enhancement of any low quality calls, audio, etc. while using nearly no memory.
* Restoring datasets: NovaSR can enhance audio quality of any audio dataset. 


### Comparisons

Comparisons were done on A100 gpu. Higher realtime means faster processing speeds.

| Model         | Speed (Real-Time) | Model Size |
| :------------ | :---------------- | :--------- |
| **NovaSR** | **3600x realtime** | **~52 KB** |
| FlowHigh      | 20x realtime        | ~450 MB     |
| FlashSR       | 14x realtime        | ~1000 MB     |
| AudioSR       | 0.6x realtime    | ~2000 MB     |

### Examples
Please check the [huggingface model](https://huggingface.co/YatharthS/NovaSR) for a few examples. 



### Usage
You can try it on [huggingface spaces](https://huggingface.co/spaces/YatharthS/NovaSR) or locally.


Simple 1 line installation:
```
pip install git+https://github.com/ysharma3501/NovaSR.git
```

Load model
```python
from NovaSR import FastSR

upsampler = FastSR() ## downloads from hf

## Use this instead for CPUs as it leads to 3-4x speedup.
# upsampler = FastSR(half=False)
```

Run model
```python
from IPython.display import Audio

## replace audio_path.wav with your wav/mp3 file
lowres_audio = upsampler.load_audio('audio_path.wav') 

## infer with model
highres_audio = upsampler.infer(lowres_audio).cpu()

display(Audio(highres_audio, rate=48000))
```
### MiniMax TTS Integration

NovaSR pairs naturally with [MiniMax TTS](https://platform.minimax.io/docs/api-reference/speech-t2a-http) to build a high-quality, low-cost text-to-speech pipeline — MiniMax generates the speech, NovaSR enhances it to 48 kHz with near-zero overhead.

**Install extra dependency:**
```
pip install requests
```

**Generate and enhance in one call:**
```python
from NovaSR import FastSR
from NovaSR.minimax_tts import generate_and_enhance
import torchaudio
import os

os.environ["MINIMAX_API_KEY"] = "your_minimax_api_key"

upsampler = FastSR()
enhanced = generate_and_enhance("Hello, world!", upsampler=upsampler)
torchaudio.save("enhanced.wav", enhanced.cpu(), 48000)
```

**Or call the steps separately:**
```python
from NovaSR.minimax_tts import generate_speech, MINIMAX_TTS_VOICES

# Step 1 — generate MP3 bytes from MiniMax TTS
audio_bytes = generate_speech(
    "Hello, world!",
    api_key="your_minimax_api_key",
    voice="English_Graceful_Lady",   # see MINIMAX_TTS_VOICES for all options
    model="speech-2.8-hd",           # or "speech-2.8-turbo" for faster synthesis
)

# Step 2 — save and enhance with NovaSR
with open("tts_output.mp3", "wb") as f:
    f.write(audio_bytes)

from NovaSR import FastSR
upsampler = FastSR()
lowres = upsampler.load_audio("tts_output.mp3")
highres = upsampler.infer(lowres).cpu()

import torchaudio
torchaudio.save("enhanced.wav", highres, 48000)
```

Available voices: `MINIMAX_TTS_VOICES` (`English_Graceful_Lady`, `English_Insightful_Speaker`, `English_radiant_girl`, `English_Persuasive_Man`, `English_Lucky_Robot`, `English_expressive_narrator`).

### Training

Please check out the kaggle notebook for training the model further on custom datasets: https://www.kaggle.com/code/yatharthsharma888/novasr-training
### Info

Q: How much data was this trained on?

A: Just 100 hours of data(mls_sidon along with vctk)

Q: How is it so small?

A: It uses less then 10 tiny conv1d layers along with snake activations based on bigvgan for maximum quality and size.

Q: Will benchmarks come?

A: Yes, I am still training it further and will benchmark it later.

## Final Notes

Repo stars and model likes would be appreciated if found helpful, thank you.


Email: yatharthsharma3501@gmail.com
