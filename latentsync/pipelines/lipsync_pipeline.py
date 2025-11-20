# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import json
import math
import os
import shutil
import subprocess
from hashlib import sha1
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Union

import cv2
import numpy as np
import soundfile as sf
import torch
import torchvision
import tqdm
from einops import rearrange
from packaging import version
from torchvision import transforms

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..utils.util import check_ffmpeg_installed, read_audio, read_video, write_video
from ..whisper.audio2feature import Audio2Feature

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

        # Disk-backed cache for expensive video-only analysis so that multiple
        # runs and processes can reuse the same preprocessing results.
        #
        # Cache layout (per entry):
        #   <video_cache_dir>/<hash>/
        #       meta.json
        #       video_frames.npz
        #       faces.pt
        #       boxes.json
        #       affine_matrices.pt
        default_cache_dir = os.environ.get("LATENTSYNC_VIDEO_CACHE")
        if default_cache_dir is None:
            # Follow XDG cache dir if available, otherwise fallback to ~/.cache
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(
                os.path.expanduser("~"), ".cache"
            )
            default_cache_dir = os.path.join(xdg_cache_home, "latentsync", "video_analysis")

        self.video_cache_dir: str = default_cache_dir
        self.enable_video_disk_cache: bool = True
        self._video_cache_format_version: int = 1

        # Lazily create the cache directory; ignore errors to avoid breaking inference.
        try:
            os.makedirs(self.video_cache_dir, exist_ok=True)
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to create video cache directory '%s'", self.video_cache_dir)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    # ---------------------------------------------------------------------
    # Video analysis and disk cache helpers
    # ---------------------------------------------------------------------

    def _make_video_cache_key(
        self,
        video_path: str,
        height: int,
        video_fps: int,
    ) -> Dict[str, Hashable]:
        """Build a metadata key describing the video and processing settings.

        This key is serialized and hashed to determine the on-disk cache
        location and to validate that a cache entry is still valid.
        """
        abs_video_path = os.path.abspath(video_path)
        try:
            stat_result = os.stat(abs_video_path)
            file_size = stat_result.st_size
            mtime = stat_result.st_mtime
        except OSError:
            # If the file is missing, we still return a key; the caller
            # will hit a cache miss and then likely fail when trying
            # to actually read the video.
            file_size = None
            mtime = None

        key = {
            "video_path": abs_video_path,
            "file_size": file_size,
            "mtime": mtime,
            "height": int(height),
            "width": int(height),
            "video_fps": int(video_fps),
            "format_version": self._video_cache_format_version,
        }
        return key

    @staticmethod
    def _hash_cache_key(key: Dict[str, Hashable]) -> str:
        """Return a stable hash for a cache key dictionary."""
        # Use a deterministic JSON representation for hashing
        key_bytes = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha1(key_bytes).hexdigest()

    def _get_cache_entry_dir(self, key: Dict[str, Hashable]) -> Path:
        """Return the directory where this key's cache entry should live."""
        key_hash = self._hash_cache_key(key)
        return Path(self.video_cache_dir) / key_hash

    def _load_video_analysis_from_cache(
        self, key: Dict[str, Hashable]
    ) -> Optional[Dict[str, Union[np.ndarray, torch.Tensor, List]]]:
        """Try to load video analysis results from the disk cache.

        Returns:
            A dict with keys: video_frames, faces, boxes, affine_matrices
            or None if the cache entry is missing or invalid.
        """
        if not self.enable_video_disk_cache:
            return None

        entry_dir = self._get_cache_entry_dir(key)
        meta_path = entry_dir / "meta.json"
        if not meta_path.exists():
            return None

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                cached_meta = json.load(f)
        except Exception:  # pylint: disable=broad-except
            return None

        # Validate metadata: if anything differs from the expected key,
        # consider the cache entry stale.
        for k, v in key.items():
            if cached_meta.get(k) != v:
                return None

        try:
            video_npz_path = entry_dir / "video_frames.npz"
            faces_path = entry_dir / "faces.pt"
            boxes_path = entry_dir / "boxes.json"
            affine_path = entry_dir / "affine_matrices.pt"

            if not (video_npz_path.exists() and faces_path.exists() and boxes_path.exists() and affine_path.exists()):
                return None

            with np.load(video_npz_path, allow_pickle=False) as npz_file:
                video_frames = npz_file["video_frames"]

            faces = torch.load(faces_path, map_location="cpu")
            with boxes_path.open("r", encoding="utf-8") as f:
                boxes = json.load(f)
            affine_matrices = torch.load(affine_path, map_location="cpu")

            logger.info("Loaded video analysis from cache: %s", entry_dir)
            return {
                "video_frames": video_frames,
                "faces": faces,
                "boxes": boxes,
                "affine_matrices": affine_matrices,
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load video analysis cache from '%s': %s", entry_dir, exc)
            return None

    def _save_video_analysis_to_cache(
        self,
        key: Dict[str, Hashable],
        video_frames: np.ndarray,
        faces: torch.Tensor,
        boxes: List,
        affine_matrices: List,
    ) -> None:
        """Persist video analysis results to disk cache."""
        if not self.enable_video_disk_cache:
            return

        entry_dir = self._get_cache_entry_dir(key)

        try:
            entry_dir.mkdir(parents=True, exist_ok=True)
            meta_path = entry_dir / "meta.json"
            video_npz_path = entry_dir / "video_frames.npz"
            faces_path = entry_dir / "faces.pt"
            boxes_path = entry_dir / "boxes.json"
            affine_path = entry_dir / "affine_matrices.pt"

            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(key, f)

            # Save video frames as compressed NumPy array
            np.savez_compressed(video_npz_path, video_frames=video_frames)

            # Faces and affine matrices are PyTorch tensors; serialize with torch.save
            torch.save(faces, faces_path)
            torch.save(affine_matrices, affine_path)

            # Boxes are a simple list of coordinates
            with boxes_path.open("w", encoding="utf-8") as f:
                json.dump(boxes, f)

            logger.info("Saved video analysis to cache: %s", entry_dir)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to save video analysis cache to '%s': %s", entry_dir, exc)

    def _compute_video_analysis(
        self,
        video_path: str,
        height: int,
        video_fps: int,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor, List]]:
        """Compute video-only analysis for a given video.

        This method performs the expensive steps:
          * Reading and converting the video to a fixed FPS.
          * Running face detection and affine alignment per frame.
        """
        # Read video frames at the target FPS (25 by default in read_video)
        video_frames = read_video(video_path, use_decord=False)

        # For the base analysis we always analyze the full video.
        faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return {
            "video_frames": video_frames,
            "faces": faces,
            "boxes": boxes,
            "affine_matrices": affine_matrices,
        }

    def _get_or_compute_video_analysis(
        self,
        video_path: str,
        height: int,
        video_fps: int,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor, List]]:
        """Get video analysis either from disk cache or by computing it."""
        key = self._make_video_cache_key(video_path, height, video_fps)

        if self.enable_video_disk_cache:
            cached = self._load_video_analysis_from_cache(key)
            if cached is not None:
                return cached

        analysis = self._compute_video_analysis(video_path, height, video_fps)
        self._save_video_analysis_to_cache(
            key,
            analysis["video_frames"],
            analysis["faces"],
            analysis["boxes"],
            analysis["affine_matrices"],
        )
        return analysis

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def loop_video(
        self,
        whisper_chunks: List,
        base_video_frames: np.ndarray,
        base_faces: torch.Tensor,
        base_boxes: List,
        base_affine_matrices: List,
    ):
        """Loop or truncate base video analysis to match the audio length.

        The original implementation recomputed face alignment on the fly
        every time. Here we instead reuse precomputed base analysis and
        build the final sequences using index mappings, so repeated runs
        with the same video can reuse disk-cached results.
        """
        num_audio = len(whisper_chunks)
        num_video = len(base_video_frames)

        if num_video == 0 or num_audio == 0:
            raise ValueError("Video frames and whisper chunks must both be non-empty.")

        # If the audio is longer than the video, we need to loop the video.
        # We follow the same pattern as the original implementation: forward
        # sequence, then reversed, repeated as needed.
        if num_audio > num_video:
            num_loops = math.ceil(num_audio / num_video)
            index_map: List[int] = []
            base_indices = list(range(num_video))
            rev_indices = base_indices[::-1]

            for i in range(num_loops):
                if i % 2 == 0:
                    index_map.extend(base_indices)
                else:
                    index_map.extend(rev_indices)

            index_map = index_map[:num_audio]
        else:
            # Audio is shorter; just take the first num_audio frames.
            index_map = list(range(num_audio))

        # Build looped/truncated sequences using the index map.
        index_tensor = torch.tensor(index_map, dtype=torch.long)

        video_frames = base_video_frames[index_map]
        faces = base_faces.index_select(0, index_tensor)
        boxes = [base_boxes[i] for i in index_map]
        affine_matrices = [base_affine_matrices[i] for i in index_map]

        return video_frames, faces, boxes, affine_matrices

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)

        # 5. Video analysis (with disk-backed caching)
        video_analysis = self._get_or_compute_video_analysis(video_path, height, video_fps)
        base_video_frames = video_analysis["video_frames"]
        base_faces = video_analysis["faces"]
        base_boxes = video_analysis["boxes"]
        base_affine_matrices = video_analysis["affine_matrices"]

        video_frames, faces, boxes, affine_matrices = self.loop_video(
            whisper_chunks, base_video_frames, base_faces, base_boxes, base_affine_matrices
        )

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)

        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
