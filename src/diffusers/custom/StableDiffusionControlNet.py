from StableDiffusionUtils import *
from CodeFormerFace import CodeFormerFace
import glob
import logging

class StableDiffusionControlNet():
  def __init__(self,modelId,pathPoses,deviceDefault = 'cpu',improveFaces = True):
    super().__init__()

    self.pathPoses = pathPoses
    self.listPosesImage = glob.glob(os.path.join(pathPoses,'*.*'),recursive = True)
    
    self.pipe = loadModelControlNet(modelId, device = deviceDefault)

    if improveFaces:
      self.modelFace = CodeFormerFace()
    else:
      self.modelFace = None

  def create(self,prompt,nimages = 1, num_inference_steps = 10, seed = SEED):

    logging.info('Image creation...')
    imgs, poses = loopInferenceControlNet(self.pipe, prompt, nimages, seed = seed, pathOpenPosePhotos = self.listPosesImage, num_inference_steps = num_inference_steps)
    logging.info('Image creation... done')

    modelFace : CodeFormerFace = self.modelFace

    if modelFace is not None:
      logging.info('Face improvement...')
      imgs = modelFace.faceImprove(imgs)
      logging.info('Face improvement... done')

    img = mergeImages(imgs+poses)
    return img