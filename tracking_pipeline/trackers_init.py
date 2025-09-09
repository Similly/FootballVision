from pathlib import Path
from boxmot import BotSort, BoostTrack

from .config import TARGET_CLASSES

def build_trackers(device, mot_device, fps):
    trackers_dict = {}
    for class_id in TARGET_CLASSES:
        if class_id == 0:  # Player
            if device.type == 'cuda':
                trackers_dict[class_id] = BotSort(
                    reid_weights=Path('reid/osnet_x1_0-stripped.pth'),
                    device=mot_device,
                    half=True,
                    track_high_thresh=0.5,
                    track_low_thresh=0.1,
                    new_track_thresh=0.6,
                    track_buffer=50,
                    frame_rate=fps,
                    proximity_thresh=0.4,
                    appearance_thresh=0.15,
                )
            else:
                trackers_dict[class_id] = BotSort(
                    with_reid=False,
                    reid_weights=None,
                    device=mot_device,
                    half=False,
                    track_high_thresh=0.5,
                    track_low_thresh=0.1,
                    new_track_thresh=0.6,
                    track_buffer=50,
                    frame_rate=fps,
                    proximity_thresh=0.4,
                )
        elif class_id == 2:  # Ball
            trackers_dict[class_id] = BoostTrack(
                with_reid=False,
                reid_weights=None,
                device=mot_device,
                half=False,
                min_hits=0,
                det_thresh=0.05,
                iou_threshold=0.00,
                min_box_area=1,
            )
        elif class_id in [1, 3]:  # GK or Ref
            trackers_dict[class_id] = BotSort(
                with_reid=False,
                reid_weights=None,
                device=mot_device,
                half=False,
                track_buffer=50,
                frame_rate=fps,
                new_track_thresh=0.6,
            )
        else:
            print(f"Unbekannte Klasse {class_id}, Tracker nicht initialisiert.")
        print(f"Tracker für Klasse {class_id} initialisiert.")
    return trackers_dict
