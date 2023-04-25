from CodeFormerUtils import *


class CodeFormerFace():
  def __init__(self,face_upsample = False, bg_upsamplerName = '', upscale = 2, detection_model = 'retinaface_resnet50', device = 'cuda'):
    super().__init__()

    self.device = device
    self.upscale = upscale

    self.net, self.face_helper, self.face_upsampler, self.bg_upsampler = initNetCodeFormer(face_upsample = face_upsample, bg_upsamplerName = bg_upsamplerName, upscale = upscale, detection_model = detection_model, device = device)


  def faceImprove(self, imgs, fidelity_weight=0.5,):

    parsedImg = doRestoreFaces(self.net, self.face_helper, self.face_upsampler, self.bg_upsampler, imgs, fidelity_weight = fidelity_weight, device=self.device, upscale=self.upscale)

    return parsedImg