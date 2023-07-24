import functools
import time
import os
import cv2
import haiku as hk
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from tqdm import tqdm

# import tree

from tapnet import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

checkpoint_path = 'tapnet/checkpoints/causal_tapir_checkpoint.npy'
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']


def build_online_model_init(frames, query_points):
    """Initialize query features for the query points."""
    model = tapir_model.TAPIR(use_causal_conv=True)

    feature_grids = model.get_feature_grids(frames, is_training=False)
    query_features = model.get_query_features(
        frames,
        is_training=False,
        query_points=query_points,
        feature_grids=feature_grids,
    )
    return query_features


def build_online_model_predict(frames, query_features, causal_context):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(use_causal_conv=True)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=query_features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True,
    )
    causal_context = trajectories['causal_context']
    del trajectories['causal_context']
    return {k: v[-1] for k, v in trajectories.items()}, causal_context


online_init = hk.transform_with_state(build_online_model_init)
online_init_apply = jax.jit(online_init.apply)

online_predict = hk.transform_with_state(build_online_model_predict)
online_predict_apply = jax.jit(online_predict.apply)

rng = jax.random.PRNGKey(42)
online_init_apply = functools.partial(
    online_init_apply, params=params, state=state, rng=rng
)
online_predict_apply = functools.partial(
    online_predict_apply, params=params, state=state, rng=rng
)


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    # visibles = occlusions < 0
    pred_occ = jax.nn.sigmoid(occlusions)
    pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
    visibles = pred_occ < 0.5  # threshold
    return visibles


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


def construct_initial_causal_state(num_points, num_resolutions):
    value_shapes = {
        "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
    }
    fake_ret = {
        k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()
    }
    return [fake_ret] * num_resolutions * 4


video = media.read_video('tapnet/examplar_videos/2086cf6f566e4c3acb2f1ecf546f833e.mp4')
video = video[::20, ...]
# media.show_video(video, fps=10)

# for img in video:
#     cv2.imshow("img", img[..., ::-1])
#     cv2.waitKey(30)

resize_height = 64  # @param {type: "integer"}
resize_width = 64  # @param {type: "integer"}
num_points = 100  # @param {type: "integer"}

height, width = video.shape[1:3]
frames = media.resize_video(video, (resize_height, resize_width))
query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)

query_features, _ = online_init_apply(frames=preprocess_frames(frames[None, None, 0]), query_points=query_points[None])
causal_state = construct_initial_causal_state(query_points.shape[0], len(query_features.resolutions) - 1)

# Predict point tracks frame by frame
predictions = []
for i in tqdm(range(frames.shape[0])):
    start = time.time()
    (prediction, causal_state), _ = online_predict_apply(
        frames=preprocess_frames(frames[None, None, i]),
        query_features=query_features,
        causal_context=causal_state,
    )
    print(time.time()-start)
    predictions.append(prediction)

tracks = np.concatenate([x['tracks'][0] for x in predictions], axis=1)
occlusions = np.concatenate([x['occlusion'][0] for x in predictions], axis=1)
expected_dist = np.concatenate([x['expected_dist'][0] for x in predictions], axis=1)

visibles = postprocess_occlusions(occlusions, expected_dist)

# Visualize sparse point tracks
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
video_viz = viz_utils.paint_point_track(video, tracks, visibles)
# media.show_video(video_viz, fps=10)
for img in video_viz:
    cv2.imshow("img", img[..., ::-1])
    cv2.waitKey(30)
