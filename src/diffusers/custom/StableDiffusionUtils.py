import gc
import os
import io
import torch
import math
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DEISMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import random

from OpenPoseUtils import createOpenPoseImage

NEGATIVE_PROMPT = 'monochrome, lowres, bad anatomy, worst quality, low quality, \
((mutilated)), extra fingers, \
mutated hands, (((mutation))), (((deformed))), ((bad anatomy)), \
(((bad proportions))), ((extra limbs)),  (((disfigured))), \
extra limbs, (bad anatomy), gross proportions, (malformed limbs), \
((missing arms)), ((missing legs)), \
(((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers)'
NEGATIVE_PROMPT = 'monochrome, lowres, bad anatomy, deformed face'

#openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
DEVICE = 'cpu'
MODEL_TYPE_RUNWAY_15 = '1.5'
MODEL_TYPE_RUNWAY_21 = '2.1'
SEED = None  # 23
GUIDANCE_SCALE = 18.5
INFERENCE_STEP = 50


def cleanMemory():
    try:
        del pipe
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def getGenerator(seed=None):
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    else:
        generator = None
    return generator


def loadModelInpainting(model_id):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16)

    pipe = pipe.to(DEVICE)

    return pipe


def loadModelText2Img(model_id, modelType=MODEL_TYPE_RUNWAY_15, device=DEVICE):

    if modelType == MODEL_TYPE_RUNWAY_21:

        if device == 'cuda':
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id)
            
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        pipe = pipe.to(device)

        if device == 'cuda':
            # this to avoid GPU out of memory:
            pipe.enable_attention_slicing()
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)

    return pipe


def loadModelImg2Img(model_id, modelType=MODEL_TYPE_RUNWAY_15, device=DEVICE):
    #model_id = "/kaggle/working/results"
    #model_id = "stabilityai/stable-diffusion-2-1"
    if modelType == MODEL_TYPE_RUNWAY_21:

        if device == 'cuda':
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id)

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead                
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        pipe = pipe.to(device)

        if device == 'cuda':
            # this to avoid GPU out of memory:
            pipe.enable_attention_slicing()
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)

    return pipe


def loadModelControlNet(model_id, modelType=MODEL_TYPE_RUNWAY_15, device=DEVICE):

    if modelType == MODEL_TYPE_RUNWAY_21:
        print('load model 2.1')

        if device == 'cuda':
            controlnet = [
                ControlNetModel.from_pretrained(
                    "thibaud/controlnet-sd21-openpose-diffusers", torch_dtype=torch.float16)
            ]
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id, controlnet=controlnet, torch_dtype=torch.float16
            )
        else:
            controlnet = [
                ControlNetModel.from_pretrained(
                    "thibaud/controlnet-sd21-openpose-diffusers").to(device)
            ]

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id, controlnet=controlnet
            )
        #pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config)

    else:
        print('load model 1.5')

        if device == 'cuda':
          controlnet = [
              ControlNetModel.from_pretrained(
                  "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
          ]
          pipe = StableDiffusionControlNetPipeline.from_pretrained(
              model_id, controlnet=controlnet, torch_dtype=torch.float16
          )
        else:
          controlnet = [
              ControlNetModel.from_pretrained(
                  "lllyasviel/sd-controlnet-openpose").to(device)
          ]

          pipe = StableDiffusionControlNetPipeline.from_pretrained(
              model_id, controlnet=controlnet
          )
        pipe.scheduler = DEISMultistepScheduler.from_config(
            pipe.scheduler.config)
        #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

    if device == 'cuda':
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

    return pipe


def mergeImages(imgs, cols=3):

    numImages = len(imgs)

    if numImages <= 0:
        print('No images found')
        return

    if numImages < cols:
        cols = numImages
    rows = math.ceil(numImages / cols)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


def doInferenceInpainting(pipe,
                          prompt,
                          image,
                          mask_image,
                          n_images=1,
                          guidance_scale=GUIDANCE_SCALE,
                          seed=None):
    imgs = []

    generator = getGenerator(seed)

    for ind in range(n_images):
        image = image.convert("RGB").resize((512, 512))
        mask_image = mask_image.convert("RGB").resize((512, 512))

        imgs.append(pipe(prompt=prompt,
                         image=image,
                         generator=generator,
                         mask_image=mask_image,
                         guidance_scale=guidance_scale).images[0])

    return imgs


def doInferenceText2Img(pipe,
                        prompt,
                        n_images=1,
                        num_inference_steps=INFERENCE_STEP,
                        guidance_scale=GUIDANCE_SCALE,
                        seed=None):
    imgs = []
    generator = getGenerator(seed)

    for _ in range(n_images):
        imgs.append(pipe(prompt,
                         generator=generator,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale).images[0])

    return imgs


def loopInferenceControlNet(pipe,
                            prompt,
                            n_images=1,
                            pathOpenPosePhotos=[],
                            num_inference_steps=INFERENCE_STEP,
                            guidance_scale=GUIDANCE_SCALE,
                            seed=None,
                            poseWeight=None):

    resImg = []
    resPoses = []

    if seed is not None:
        random.seed(seed)

    for _ in range(n_images):
        fullPath = random.choice(pathOpenPosePhotos)

        openposeImage = createOpenPoseImage(fullPath)
        resPoses.append(openposeImage)

        resImage = doInferenceControlNet(pipe,
                                         prompt,
                                         n_images=1,
                                         num_inference_steps=num_inference_steps,
                                         guidance_scale=guidance_scale,
                                         seed=seed,
                                         imagesRef=[openposeImage],
                                         poseWeight=poseWeight)

        resImg += resImage

    return resImg, resPoses


def doInferenceControlNet(pipe,
                          prompt,
                          n_images=1,
                          num_inference_steps=INFERENCE_STEP,
                          guidance_scale=GUIDANCE_SCALE,
                          seed=None,
                          imagesRef=[],
                          poseWeight=None,
                          device=DEVICE):
    imgs = []

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = torch.Generator(device=device)

    if poseWeight is None:
        controlnet_conditioning_scale = [1.0, 0.8]
    else:
        controlnet_conditioning_scale = poseWeight

    for ind in range(n_images):
        imgs.append(pipe(
            prompt,
            imagesRef,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale
        ).images[0])

    return imgs


def doInferenceImg2Img(pipe,
                       prompt,
                       init_image,
                       n_images=1,
                       num_inference_steps=INFERENCE_STEP,
                       guidance_scale=GUIDANCE_SCALE,
                       strength=0.75,
                       seed=None,
                       generator = None):
    imgs = []

    if generator is None:
        generator = getGenerator(seed)

    for ind in range(n_images):
        imgs.append(pipe(prompt=prompt,
                         generator=generator,
                         image=init_image,
                         num_inference_steps=num_inference_steps,
                         strength=strength,
                         guidance_scale=guidance_scale).images[0])

    return imgs


def loopInferenceImg2Img(pipe,
                         prompt,
                         init_image,
                         n_images=2,
                         num_inference_steps=50,
                         seed=None):
    allimages = []
    generator = getGenerator(seed)

    counter = 0
    for indguidance in range(0, 5, 1):
        indguidance = indguidance / 5 * 19 + 1
        for indstrength in range(0, 10, 2):
            indstrength = indstrength / 10
            print('row: ' + str(counter) + ' guidance: ' +
                  str(indguidance) + ' strength: ' + str(indstrength))
            imgs = doInferenceImg2Img(pipe,
                                      prompt,
                                      num_inference_steps=num_inference_steps,
                                      init_image=init_image,
                                      n_images=n_images,
                                      guidance_scale=indguidance,
                                      strength=indstrength,
                                      seed=seed,
                                      generator=generator)
            allimages += imgs
            counter += 1

    return allimages


def openImageFromFile(pathImage):
    return Image.open(pathImage).convert("RGB")


def doInferenceText2Img(pipe,
                        prompt,
                        n_images=1,
                        num_inference_steps=INFERENCE_STEP,
                        guidance_scale=GUIDANCE_SCALE,
                        seed=None):
    imgs = []
    generator = getGenerator(seed)

    for _ in range(n_images):
        imgs.append(pipe(prompt,
                         generator=generator,
                         num_inference_steps=num_inference_steps,
                         negative_prompt=NEGATIVE_PROMPT,
                         guidance_scale=guidance_scale).images[0])

    return imgs
