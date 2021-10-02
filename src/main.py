from src.config.config import MyConfig
from src.controllers.faceController import FaceController

if __name__ == '__main__':
    face = FaceController(MyConfig.imgOriginalPath)
