#!/usr/bin/env python3
# @daryltucker

import torch

import os
import time
import re
import random

import node_helpers
import numpy as np

from PIL import Image, ImageSequence, ImageOps


category = "LoadFiles"


GENERIC_INPUT_TYPES = {
  "required": {
    "directory": ("STRING", {"multiline": False, "default": "output/"},),
    "limiter": ("STRING", {"multiline": False, "default": ".*.png"},),
    "sort": (["Name", "Date Created", "Date Modified", "Size"],),
    "direction": (["Acending", "Decending"],),
    "splice": (['Tail', 'Head'],),
    "skip": ("INT", {"default": 1, "min": 0},),
    "count": ("INT", {"default": 1, "min": 0},),
    "error": (["No Error", "Load Count"],),
  }
}


def filterFn(option):
    filter_key = os.path.basename
    match option:
        case 'Name':
            filter_key = os.path.basename
        case 'Date Created':
            filter_key = os.path.getctime
        case 'Date Modified':
            filter_key = os.path.getmtime
        case 'Size':
            filter_key = os.path.getsize

    return filter_key


def limiterFn(file, directory, limiter):
    fullpath = f"{directory}{file}"
    results = False
    try:
        results = (os.path.isfile(fullpath) and re.search(limiter, file))
    except re.error as e:
        raise Exception("LoadFiles: Double-check your regex. EXAMPLE Valid: '.*' Invalid: '*.*'")
        print(e)
    return results


def countCheck(files, count, error):
    # Determine if Count has been met
    if len(files) < count:
        msg = f"LoadFiles: The directory has fewer than {count} files."
        if error == "Load Count":
            raise Exception(msg)
        else:
            print(msg)

        # Fill Files List to fulfil Count request.
        if len(files) > 0:
            diff = count - len(files)
            for i in range(0, diff):
                files.append(files[i])

    return files


def spliceFiles(files, count, splice, skip):
    files_txt = ''
    if files and count:
        if splice == "Head":
            files = files[skip:skip + count]
        else:
            files = files[-1 * count - skip:][:-1 * skip]
    files_txt = '\n'.join(files)

    return (files, files_txt)


# LoadImages #################################################################

class LoadImages:

    @classmethod
    def INPUT_TYPES(cls):
        return GENERIC_INPUT_TYPES

    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    FUNCTION = "listFiles"
    CATEGORY = category

    def listFiles(self, directory, limiter, sort, direction, splice, count, error):
        count = int(count)

        if directory[-1:] != '/':
            directory = f"{directory}/"
        if not os.path.isdir(directory):
            raise Exception(f"LoadFiles: The path is not a valid directory: {directory}")

        filter_key = filterFn(sort)

        files = sorted(
            filter(
                lambda file: limiterFn(file, directory, limiter),
                os.listdir(directory)
            ),
            reverse=True if direction == "Decending" else False,
            key=lambda file: filter_key(f"{directory}{file}")
        )

        # Determine if Count has been met
        files = countCheck(files, count, error)
        # Splice Files
        files, files_txt = spliceFiles(files, count, splice)

        # Process Images to gen Masks
        output_images = []
        output_masks = []
        w, h = None, None

        for filename in files:
            filename = f"{directory}{filename}"
            img = node_helpers.pillow(Image.open, filename)

            excluded_formats = ['MPO']

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                if len(output_images) == 0:
                    w = image.size[0]
                    h = image.size[1]

                if image.size[0] != w or image.size[1] != h:
                    continue

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, files_txt)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Always refresh files
        return time.time()


# ListFilenames ##############################################################
class ListFilenames:

    @classmethod
    def INPUT_TYPES(cls):
        return GENERIC_INPUT_TYPES

    RETURN_TYPES = ("STRING",)
    FUNCTION = "listFiles"
    CATEGORY = category

    def listFiles(self, directory, limiter, sort, direction, splice, count, error):
        count = int(count)

        if directory[-1:] != '/':
            directory = f"{directory}/"
        if not os.path.isdir(directory):
            raise Exception(f"LoadFiles: The path is not a valid directory: {directory}")

        filter_key = filterFn(sort)

        files = sorted(
            filter(
                lambda file: limiterFn(file, directory, limiter),
                os.listdir(directory)
            ),
            reverse=True if direction == "Decending" else False,
            key=lambda file: filter_key(f"{directory}{file}")
        )

        # Determine if Count has been met
        files = countCheck(files, count, error)
        # Splice Files
        files, files_txt = spliceFiles(files, count, splice)

        return (files_txt,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Always refresh files
        return time.time()


# CountLines #################################################################
class CountLines:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "STRING": ("STRING", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    FUNCTION = "countLines"
    CATEGORY = category

    def countLines(self, STRING):
        count = len(re.findall('\r*\n', STRING))
        if STRING:
            count = count + 1
        return (int(count), float(count), str(count),)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Always refresh files
        return time.time()


# ComfyUI Mappings ###########################################################

NODE_CLASS_MAPPINGS = {
    "LoadImages": LoadImages,
    "ListFilenames": ListFilenames,
    "CountLines": CountLines
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImages": "Load Images from Directory",
    "ListFilenames": "List Files in Directory",
    "CountLines": "Line Count STRING"
}
