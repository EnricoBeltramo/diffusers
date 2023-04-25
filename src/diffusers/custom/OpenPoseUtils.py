from controlnet_aux import OpenposeDetector

from diffusers.utils import load_image

openpose = None

def getOpenPoseImage(model = "lllyasviel/ControlNet"):
    global openpose
    if openpose is None:
      openpose = OpenposeDetector.from_pretrained(model)
    return openpose

def createOpenPoseImage(pathImage):
    openpose = getOpenPoseImage()
    openpose_image = load_image(pathImage)
    return openpose(openpose_image)
