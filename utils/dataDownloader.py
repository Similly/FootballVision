from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

# Zielverzeichnis für die Daten
sn = SoccerNetDownloader(LocalDirectory="data")

sn.password = "s0cc3rn3t"

# Herunterladen der Tracking-Daten für verschiedene Splits
#mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
#sn.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
sn.downloadRAWVideo(dataset="SoccerNet-Tracking")

#print(getListGames(split="v1")) # return list of games from SoccerNetv1 (train/valid/test)