import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS

import_custom_nodes()

(
    unetloader_12,
    dualcliploader_11,
    vaeloader_10,
    checkpointloadersimple_316,
    reactorfaceboost_241,
    bboxdetectorloaderfaceparsing_303,
    lora_loader_stack_rgthree_238,
    lora_loader_stack_rgthree_278,
    loraloadermodelonly_323,
    cliptextencode,
    ksamplerselect,
    randomnoise,
    emptylatentimage,
    ksampler,
    vaedecode,
    loadimage,
    reactorfaceswap,
    reactormaskhelper,
    vaeencode,
    basicguider,
    basicscheduler,
    splitsigmas,
    samplercustomadvanced,
    bboxdetectfaceparsing,
    bboxlistitemselectfaceparsing,
    imagecropwithbboxfaceparsing,
    imageresizecalculatorfaceparsing,
    imagescaleby,
    cliptextencodesdxl,
    denseposepreprocessor,
    imagetomask,
    tobinarymask,
    impactdilatemask,
    impactgaussianblurmask,
    inpaintmodelconditioning,
    imagesizefaceparsing,
    emptyimage,
    imagepadforoutpaint,
    masktoimage,
    imagescale,
    invertmask,
    imageinsertwithbboxfaceparsing,
    imagecompositemasked,
    saveimage,
) = [None] * 44


def init_foundaton_nodes():
    global unetloader_12
    global dualcliploader_11
    global vaeloader_10
    global checkpointloadersimple_316
    global reactorfaceboost_241
    global bboxdetectorloaderfaceparsing_303
    global cliptextencode, ksamplerselect, randomnoise, emptylatentimage, ksampler, vaedecode, loadimage, reactorfaceswap, reactormaskhelper, vaeencode, basicguider, basicscheduler, splitsigmas, samplercustomadvanced, bboxdetectfaceparsing, bboxlistitemselectfaceparsing, imagecropwithbboxfaceparsing, imageresizecalculatorfaceparsing, imagescaleby, cliptextencodesdxl, denseposepreprocessor, imagetomask, tobinarymask, impactdilatemask, impactgaussianblurmask, inpaintmodelconditioning, imagesizefaceparsing, emptyimage, imagepadforoutpaint, masktoimage, imagescale, invertmask, imageinsertwithbboxfaceparsing, imagecompositemasked, saveimage
    unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
    unetloader_12 = unetloader.load_unet(
        unet_name="flux1-dev.safetensors", weight_dtype="default"
    )

    dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
    dualcliploader_11 = dualcliploader.load_clip(
        clip_name1="t5xxl_fp16.safetensors",
        clip_name2="clip_l.safetensors",
        type="flux",
    )

    vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
    vaeloader_10 = vaeloader.load_vae(vae_name="ae.safetensors")

    checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    checkpointloadersimple_316 = checkpointloadersimple.load_checkpoint(
        ckpt_name="lustify.safetensors"
    )

    reactorfaceboost = NODE_CLASS_MAPPINGS["ReActorFaceBoost"]()
    reactorfaceboost_241 = reactorfaceboost.execute(
        enabled=True,
        boost_model="GFPGANv1.4.pth",
        interpolation="Bicubic",
        visibility=1,
        codeformer_weight=0.5,
        restore_with_main_after=False,
    )

    bboxdetectorloaderfaceparsing = NODE_CLASS_MAPPINGS[
        "BBoxDetectorLoader(FaceParsing)"
    ]()
    bboxdetectorloaderfaceparsing_303 = bboxdetectorloaderfaceparsing.main(
        model_name="bbox/face_yolov8m.pt"
    )

    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
    randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
    emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    reactorfaceswap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
    reactormaskhelper = NODE_CLASS_MAPPINGS["ReActorMaskHelper"]()
    vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
    basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
    basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
    splitsigmas = NODE_CLASS_MAPPINGS["SplitSigmas"]()
    samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
    bboxdetectfaceparsing = NODE_CLASS_MAPPINGS["BBoxDetect(FaceParsing)"]()
    bboxlistitemselectfaceparsing = NODE_CLASS_MAPPINGS[
        "BBoxListItemSelect(FaceParsing)"
    ]()
    imagecropwithbboxfaceparsing = NODE_CLASS_MAPPINGS[
        "ImageCropWithBBox(FaceParsing)"
    ]()
    imageresizecalculatorfaceparsing = NODE_CLASS_MAPPINGS[
        "ImageResizeCalculator(FaceParsing)"
    ]()
    imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
    cliptextencodesdxl = NODE_CLASS_MAPPINGS["CLIPTextEncodeSDXL"]()
    denseposepreprocessor = NODE_CLASS_MAPPINGS["DensePosePreprocessor"]()
    imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
    tobinarymask = NODE_CLASS_MAPPINGS["ToBinaryMask"]()
    impactdilatemask = NODE_CLASS_MAPPINGS["ImpactDilateMask"]()
    impactgaussianblurmask = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]()
    inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
    imagesizefaceparsing = NODE_CLASS_MAPPINGS["ImageSize(FaceParsing)"]()
    emptyimage = NODE_CLASS_MAPPINGS["EmptyImage"]()
    imagepadforoutpaint = NODE_CLASS_MAPPINGS["ImagePadForOutpaint"]()
    masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
    imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
    invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
    imageinsertwithbboxfaceparsing = NODE_CLASS_MAPPINGS[
        "ImageInsertWithBBox(FaceParsing)"
    ]()
    imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()


def init_specific_nodes(job_id: str):
    global lora_loader_stack_rgthree_238
    global lora_loader_stack_rgthree_278
    global loraloadermodelonly_323

    lora_loader_stack_rgthree = NODE_CLASS_MAPPINGS["Lora Loader Stack (rgthree)"]()
    lora_loader_stack_rgthree_238 = lora_loader_stack_rgthree.load_lora(
        lora_01="nsfw_FLUXTASTIC_lora.safetensors",
        strength_01=1,
        lora_02=f"flux_{job_id}.safetensors",
        strength_02=1,
        lora_03="None",
        strength_03=1,
        lora_04="None",
        strength_04=1,
        model=get_value_at_index(unetloader_12, 0),
        clip=get_value_at_index(dualcliploader_11, 0),
    )
    lora_loader_stack_rgthree_278 = lora_loader_stack_rgthree.load_lora(
        lora_01="None",
        strength_01=1,
        lora_02=f"flux_{job_id}.safetensors",
        strength_02=1,
        lora_03="None",
        strength_03=1,
        lora_04="None",
        strength_04=1,
        model=get_value_at_index(unetloader_12, 0),
        clip=get_value_at_index(dualcliploader_11, 0),
    )

    loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
    loraloadermodelonly_323 = loraloadermodelonly.load_lora_model_only(
        lora_name=f"sdxl_{job_id}.safetensors",
        strength_model=1.1,
        model=get_value_at_index(checkpointloadersimple_316, 0),
    )


def main(
    job_id: str,
    prompt: str,
    width: int = 768,
    height: int = 1024,
):
    print("X" * 50)
    print("Starting main function")

    loadimage_304 = loadimage.load_image(image=f"{job_id}.png")

    with torch.inference_mode():
        cliptextencode_6 = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(lora_loader_stack_rgthree_238, 1),
        )

        ksamplerselect_16 = ksamplerselect.get_sampler(sampler_name="deis")

        randomnoise_50 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        cliptextencode_317 = cliptextencode.encode(
            text=f"embedding:{job_id}.pt {prompt}",
            clip=get_value_at_index(checkpointloadersimple_316, 1),
        )

        cliptextencode_318 = cliptextencode.encode(
            text="bokeh, film grain, bokeh, dreamy haze, technicolor, underexposed, low quality, lowres",
            clip=get_value_at_index(checkpointloadersimple_316, 1),
        )

        emptylatentimage_319 = emptylatentimage.generate(
            width=width, height=height, batch_size=1
        )

        ksampler_320 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=30,
            cfg=4,
            sampler_name="dpmpp_sde",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(loraloadermodelonly_323, 0),
            positive=get_value_at_index(cliptextencode_317, 0),
            negative=get_value_at_index(cliptextencode_318, 0),
            latent_image=get_value_at_index(emptylatentimage_319, 0),
        )

        vaedecode_321 = vaedecode.decode(
            samples=get_value_at_index(ksampler_320, 0),
            vae=get_value_at_index(checkpointloadersimple_316, 2),
        )

        reactorfaceswap_240 = reactorfaceswap.execute(
            enabled=True,
            swap_model="inswapper_128.onnx",
            facedetection="retinaface_resnet50",
            face_restore_model="none",
            face_restore_visibility=1,
            codeformer_weight=0.5,
            detect_gender_input="no",
            detect_gender_source="no",
            input_faces_index="0",
            source_faces_index="0",
            console_log_level=1,
            input_image=get_value_at_index(vaedecode_321, 0),
            source_image=get_value_at_index(loadimage_304, 0),
            face_boost=get_value_at_index(reactorfaceboost_241, 0),
        )

        reactormaskhelper_242 = reactormaskhelper.execute(
            bbox_model_name="bbox/face_yolov8m.pt",
            bbox_threshold=0.5,
            bbox_dilation=10,
            bbox_crop_factor=3,
            bbox_drop_size=10,
            sam_model_name="sam_vit_b_01ec64.pth",
            sam_dilation=0,
            sam_threshold=0.93,
            bbox_expansion=0,
            mask_hint_threshold=0.7,
            mask_hint_use_negative="False",
            morphology_operation="dilate",
            morphology_distance=0,
            blur_radius=9,
            sigma_factor=1,
            image=get_value_at_index(vaedecode_321, 0),
            swapped_image=get_value_at_index(reactorfaceswap_240, 0),
        )

        vaeencode_94 = vaeencode.encode(
            pixels=get_value_at_index(reactormaskhelper_242, 0),
            vae=get_value_at_index(vaeloader_10, 0),
        )

        cliptextencode_263 = cliptextencode.encode(
            text="detailed face, realistic, highres, high resolution",
            clip=get_value_at_index(lora_loader_stack_rgthree_278, 1),
        )

        ksamplerselect_269 = ksamplerselect.get_sampler(sampler_name="deis")

        randomnoise_273 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        basicguider_22 = basicguider.get_guider(
            model=get_value_at_index(lora_loader_stack_rgthree_238, 0),
            conditioning=get_value_at_index(cliptextencode_6, 0),
        )

        basicscheduler_17 = basicscheduler.get_sigmas(
            scheduler="sgm_uniform",
            steps=8,
            denoise=0.45,
            model=get_value_at_index(unetloader_12, 0),
        )

        splitsigmas_38 = splitsigmas.get_sigmas(
            step=0, sigmas=get_value_at_index(basicscheduler_17, 0)
        )

        samplercustomadvanced_13 = samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_50, 0),
            guider=get_value_at_index(basicguider_22, 0),
            sampler=get_value_at_index(ksamplerselect_16, 0),
            sigmas=get_value_at_index(splitsigmas_38, 1),
            latent_image=get_value_at_index(vaeencode_94, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_13, 0),
            vae=get_value_at_index(vaeloader_10, 0),
        )

        bboxdetectfaceparsing_302 = bboxdetectfaceparsing.main(
            threshold=0.3,
            dilation=8,
            dilation_ratio=0.2,
            by_ratio=False,
            bbox_detector=get_value_at_index(bboxdetectorloaderfaceparsing_303, 0),
            image=get_value_at_index(vaedecode_8, 0),
        )

        bboxlistitemselectfaceparsing_247 = bboxlistitemselectfaceparsing.main(
            index=0, bbox_list=get_value_at_index(bboxdetectfaceparsing_302, 0)
        )

        imagecropwithbboxfaceparsing_249 = imagecropwithbboxfaceparsing.main(
            bbox=get_value_at_index(bboxlistitemselectfaceparsing_247, 0),
            image=get_value_at_index(vaedecode_8, 0),
        )

        imageresizecalculatorfaceparsing_250 = imageresizecalculatorfaceparsing.main(
            target_size=1024,
            force_8x=False,
            force_64x=False,
            image=get_value_at_index(imagecropwithbboxfaceparsing_249, 0),
        )

        imagescaleby_251 = imagescaleby.upscale(
            upscale_method="lanczos",
            scale_by=get_value_at_index(imageresizecalculatorfaceparsing_250, 5),
            image=get_value_at_index(imagecropwithbboxfaceparsing_249, 0),
        )

        vaeencode_274 = vaeencode.encode(
            pixels=get_value_at_index(imagescaleby_251, 0),
            vae=get_value_at_index(vaeloader_10, 0),
        )

        cliptextencodesdxl_357 = cliptextencodesdxl.encode(
            width=width,
            height=height,
            crop_w=0,
            crop_h=0,
            target_width=width,
            target_height=height,
            text_g=f"embedding:{job_id}.pt {prompt}",
            text_l=f"embedding:{job_id}.pt {prompt}",
            clip=get_value_at_index(checkpointloadersimple_316, 1),
        )

        cliptextencodesdxl_358 = cliptextencodesdxl.encode(
            width=width,
            height=height,
            crop_w=0,
            crop_h=0,
            target_width=width,
            target_height=height,
            text_g="bokeh, film grain, bokeh, dreamy haze, technicolor, underexposed, low quality, lowres",
            text_l="bokeh, film grain, bokeh, dreamy haze, technicolor, underexposed, low quality, lowres",
            clip=get_value_at_index(checkpointloadersimple_316, 1),
        )

        denseposepreprocessor_332 = denseposepreprocessor.execute(
            model="densepose_r50_fpn_dl.torchscript",
            cmap="Parula (CivitAI)",
            resolution=512,
            image=get_value_at_index(vaedecode_8, 0),
        )

        imagetomask_347 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(denseposepreprocessor_332, 0)
        )

        tobinarymask_348 = tobinarymask.doit(
            threshold=1, mask=get_value_at_index(imagetomask_347, 0)
        )

        impactdilatemask_351 = impactdilatemask.doit(
            dilation=10, mask=get_value_at_index(tobinarymask_348, 0)
        )

        impactgaussianblurmask_350 = impactgaussianblurmask.doit(
            kernel_size=30, sigma=10, mask=get_value_at_index(impactdilatemask_351, 0)
        )

        inpaintmodelconditioning_356 = inpaintmodelconditioning.encode(
            positive=get_value_at_index(cliptextencodesdxl_357, 0),
            negative=get_value_at_index(cliptextencodesdxl_358, 0),
            vae=get_value_at_index(checkpointloadersimple_316, 2),
            pixels=get_value_at_index(vaedecode_8, 0),
            mask=get_value_at_index(impactgaussianblurmask_350, 0),
        )

        for q in range(1):
            imagesizefaceparsing_253 = imagesizefaceparsing.main(
                image=get_value_at_index(imagescaleby_251, 0)
            )

            emptyimage_254 = emptyimage.generate(
                width=get_value_at_index(imagesizefaceparsing_253, 0),
                height=get_value_at_index(imagesizefaceparsing_253, 1),
                batch_size=1,
                color=0,
            )

            imagepadforoutpaint_258 = imagepadforoutpaint.expand_image(
                left=96,
                top=96,
                right=96,
                bottom=96,
                feathering=128,
                image=get_value_at_index(emptyimage_254, 0),
            )

            masktoimage_259 = masktoimage.mask_to_image(
                mask=get_value_at_index(imagepadforoutpaint_258, 1)
            )

            imagescale_257 = imagescale.upscale(
                upscale_method="nearest-exact",
                width=get_value_at_index(imagesizefaceparsing_253, 0),
                height=get_value_at_index(imagesizefaceparsing_253, 1),
                crop="disabled",
                image=get_value_at_index(masktoimage_259, 0),
            )

            imagetomask_256 = imagetomask.image_to_mask(
                channel="red", image=get_value_at_index(imagescale_257, 0)
            )

            invertmask_255 = invertmask.invert(
                mask=get_value_at_index(imagetomask_256, 0)
            )

            basicguider_271 = basicguider.get_guider(
                model=get_value_at_index(lora_loader_stack_rgthree_278, 0),
                conditioning=get_value_at_index(cliptextencode_263, 0),
            )

            basicscheduler_270 = basicscheduler.get_sigmas(
                scheduler="sgm_uniform",
                steps=8,
                denoise=0.4,
                model=get_value_at_index(unetloader_12, 0),
            )

            splitsigmas_272 = splitsigmas.get_sigmas(
                step=0, sigmas=get_value_at_index(basicscheduler_270, 0)
            )

            samplercustomadvanced_268 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_273, 0),
                guider=get_value_at_index(basicguider_271, 0),
                sampler=get_value_at_index(ksamplerselect_269, 0),
                sigmas=get_value_at_index(splitsigmas_272, 1),
                latent_image=get_value_at_index(vaeencode_274, 0),
            )

            vaedecode_264 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_268, 0),
                vae=get_value_at_index(vaeloader_10, 0),
            )

            ksampler_359 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=4,
                sampler_name="dpmpp_sde",
                scheduler="karras",
                denoise=0.4,
                model=get_value_at_index(loraloadermodelonly_323, 0),
                positive=get_value_at_index(inpaintmodelconditioning_356, 0),
                negative=get_value_at_index(inpaintmodelconditioning_356, 1),
                latent_image=get_value_at_index(inpaintmodelconditioning_356, 2),
            )

            vaedecode_362 = vaedecode.decode(
                samples=get_value_at_index(ksampler_359, 0),
                vae=get_value_at_index(checkpointloadersimple_316, 2),
            )

            imagesizefaceparsing_284 = imagesizefaceparsing.main(
                image=get_value_at_index(vaedecode_362, 0)
            )

            emptyimage_283 = emptyimage.generate(
                width=get_value_at_index(imagesizefaceparsing_284, 0),
                height=get_value_at_index(imagesizefaceparsing_284, 1),
                batch_size=1,
                color=0,
            )

            imagescaleby_280 = imagescaleby.upscale(
                upscale_method="nearest-exact",
                scale_by=get_value_at_index(imageresizecalculatorfaceparsing_250, 5),
                image=get_value_at_index(vaedecode_264, 0),
            )

            imageinsertwithbboxfaceparsing_282 = imageinsertwithbboxfaceparsing.main(
                bbox=get_value_at_index(bboxlistitemselectfaceparsing_247, 0),
                image_src=get_value_at_index(emptyimage_283, 0),
                image=get_value_at_index(imagescaleby_280, 0),
            )

            masktoimage_287 = masktoimage.mask_to_image(
                mask=get_value_at_index(invertmask_255, 0)
            )

            imageinsertwithbboxfaceparsing_286 = imageinsertwithbboxfaceparsing.main(
                bbox=get_value_at_index(bboxlistitemselectfaceparsing_247, 0),
                image_src=get_value_at_index(emptyimage_283, 0),
                image=get_value_at_index(masktoimage_287, 0),
            )

            imagetomask_285 = imagetomask.image_to_mask(
                channel="red",
                image=get_value_at_index(imageinsertwithbboxfaceparsing_286, 0),
            )

            imagecompositemasked_279 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=False,
                destination=get_value_at_index(vaedecode_362, 0),
                source=get_value_at_index(imageinsertwithbboxfaceparsing_282, 0),
                mask=get_value_at_index(imagetomask_285, 0),
            )

            saveimage_382 = saveimage.save_images(
                filename_prefix=job_id + "/" + job_id,
                images=get_value_at_index(imagecompositemasked_279, 0),
            )

            return saveimage_382
