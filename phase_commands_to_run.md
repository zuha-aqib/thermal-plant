# commands to run

## generate synthetically moved accidental video
so basically it moves the video to a certain degree after x seconds

```powershell
py -3.12 .\simulate_accidental_camera_movement.py ".\raw-videos\97. A- 200 Corridor Thermal.mp4" --move-at-seconds 4 --dx 20 --dy -10 --rotation 1 
```

```powershell
py -3.12 .\simulate_accidental_camera_movement.py ".\raw-videos\95.fur incin AB Stacks Thermal.mp4" --move-at-seconds 4 --dx 20 --dy -10 --rotation 1 
```

## phase A 
it maps the annotations on the moved video by understanding the movement and then moving the annotations accordingly

```powershell
py -3.12 .\phase_a_accidental_camera_motion.py ".\synthetic-moved-videos\97. A- 200 Corridor Thermal\accidental_001.mp4" ".\step-01-annotate-videos\97. A- 200 Corridor Thermal\regions.json" --show-fixed-polygons
```

```powershell
py -3.12 .\phase_a_accidental_camera_motion.py ".\synthetic-moved-videos\95.fur incin AB Stacks Thermal\accidental_001.mp4" ".\step-01-annotate-videos\95.fur incin AB Stacks Thermal\regions.json" --show-fixed-polygons
```

## generate synthetically moving incidental video
so it basically takes two ANNOTATED videos, and moves the camera between them

```powershell
py -3.12 .\simulate_incidental_camera_movement.py ".\raw-videos\95.fur incin AB Stacks Thermal.mp4" ".\raw-videos\97. A- 200 Corridor Thermal.mp4" --return-to-first
```

```powershell
py -3.12 .\simulate_incidental_camera_movement.py ".\raw-videos\97. A- 200 Corridor Thermal.mp4" ".\raw-videos\95.fur incin AB Stacks Thermal.mp4" --return-to-first
```

## phase B
so taking the synthetically moved incidnetal video which has two views, it breaks each frame down and understands ok this is stable, this is moving, this is a new view and this is an old view etc

```powershell
py -3.12 .\phase_b_scene_change_detector.py "synthetic-incidental-videos\95.fur incin AB Stacks Thermal__TO__97. A- 200 Corridor Thermal\incidental_001.mp4"
```

```powershell
py -3.12 .\phase_b_scene_change_detector.py "synthetic-incidental-videos\97. A- 200 Corridor Thermal__TO__95.fur incin AB Stacks Thermal\incidental_001.mp4"
```

## phase C part A
so here we need to generate a dataset using the already existing videos and make frames etc

```powershell

```