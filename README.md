## NovaSR: Pushing the Limits of Extreme Efficiency in Audio Super-Resolution
This is the repository for NovaSR, a tiny 50kb audio upsampling model that upscales muffled 16khz audio into clear and crisp 48khz audio at speeds over 3500x realtime. 

### Key benefits
* Speed: Can reach 3600x realtime speed on a single a100 gpu.
* Quality: On par with models 5,000x larger.
* Size: Just 52kb in size, several thousand times smaller then most. 

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


### Usage

Simple 1 line installation:
```
pip install git+https://github.com/ysharma3501/NovaSR.git
```

Load model
```python
from NovaSR import FastSR

upsampler = FastSR() ## downloads from hf
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
