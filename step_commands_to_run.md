# commands to run

## points to remember
if python is not on PATH or multiple python versions exist, use ```py``` to access python and ```-version``` to specify the version. i have both 3.12 and 3.14 but 3.12 was more compatible with all the libraries and packages

## step 1
annotate the videos by specifying the plant box

to annotate a specific video:

```powershell
py -3.12  annotate_thermal_video.py "raw-videos/95.fur incin AB Stacks Thermal.mp4"
```

to annotate the whole raw-videos folder:

```powershell
py -3.12 .\annotate_thermal_video.py .\raw-videos 
```


## step 2
mark the scale and min max temperature and test the ocr

- if tesseract is not on PATH, specify it in the command line

```powershell
py -3.12 .\configure_thermal_scale.py ".\raw-videos\95.fur incin AB Stacks Thermal.mp4" ".\annotated-videos\95.fur incin AB Stacks Thermal\regions.json" --tesseract "C:\Users\zuha.aqib\AppData\Local\Programs\Tesseract-OCR\tesseract.exe" 
```

## step 3
apply the scale and annotations to compute the temperature at each pixel and then the ROI (average, min and max of each plant)

- if tesseract is not on PATH, specify it in the command line
- by default max scale jump (the difference of scale between two consecutive frames) is 2.5. you may change it accordingly in command line
- ```ocr-hz``` means the change of scale per second

```powershell
py -3.12 .\process_thermal_temperatures.py ".\raw-videos\95.fur incin AB Stacks Thermal.mp4" ".\annotated-videos\95.fur incin AB Stacks Thermal\regions.json" ".\thermal_scale_config\scale_config.json" --tesseract "C:\Users\zuha.aqib\AppData\Local\Programs\Tesseract-OCR\tesseract.exe" --ocr-hz 3 --max-scale-jump 2.0
```