import numpy as np
import time
import tensorflow as tf
from scipy.signal import medfilt
from repnet import ResnetPeriodEstimator

from scipy import stats


N = 64

CDF0 = (np.arange(0, N) / N).reshape((-1, 1))
CDF1 = (np.arange(1.0, N + 1) / N).reshape((-1, 1))


def get_repnet_model(logdir):
    """Returns a trained RepNet model.

  Args:
    logdir (string): Path to directory where checkpoint will be downloaded.

  Returns:
    model (Keras model): Trained RepNet model.
  """
    # Check if we are in eager mode.
    assert tf.executing_eagerly()

    # Models will be called in eval mode.
    # tf.keras.backend.set_learning_phase(0)

    # QRS
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        # tf.config.experimental.set_memory_growth(gpu, True)
        # or
        print('Limit fix memory: 15120')
        tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15120)])

    # Define RepNet model.
    model = ResnetPeriodEstimator()
    # tf.function for speed.
    model.call = tf.function(model.call, experimental_relax_shapes=True)

    # Define checkpoint and checkpoint manager.
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=logdir, max_to_keep=10)
    latest_ckpt = ckpt_manager.latest_checkpoint
    print('Loading from: ', latest_ckpt)
    if not latest_ckpt:
        raise ValueError('Path does not have a checkpoint to load.')
    # Restore weights.
    ckpt.restore(latest_ckpt).expect_partial()

    # Pass dummy frames to build graph.
    model(tf.random.uniform((1, 64, 112, 112, 3)))
    return model


def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    query_frame = (query_frame - min_v) / max(1e-7, (max_v - min_v))
    return query_frame


def get_counts(model, frames, strides, batch_size,
               threshold,
               within_period_threshold,
               constant_speed=False,
               median_filter=False,
               fully_periodic=False, osd_feat=False, pcaks=None, progress_cb=None):
    """Pass frames through model and conver period predictions to count."""
    seq_len = len(frames)
    raw_scores_list = []
    scores = []
    embs_list = []
    within_period_scores_list = []

    if fully_periodic:
        within_period_threshold = 0.0

    frames = model.preprocess(frames)

    Fprg = 1.0
    if pcaks:
        Fprg = 0.5

    for i, stride in enumerate(strides):
        num_batches = int(np.ceil(seq_len / model.num_frames / stride / batch_size))
        raw_scores_per_stride = []
        within_period_score_stride = []
        embs_stride = []
        Nprg = num_batches * len(strides)
        for batch_idx in range(num_batches):
            idxes = tf.range(batch_idx * batch_size * model.num_frames * stride,
                    (batch_idx + 1) * batch_size * model.num_frames * stride,
                    stride)
            idxes = tf.clip_by_value(idxes, 0, seq_len - 1)
            curr_frames = tf.gather(frames, idxes)
            curr_frames = tf.reshape(
                curr_frames,
                [batch_size, model.num_frames, model.image_size, model.image_size, 3])

            raw_scores, within_period_scores, embs = model(curr_frames)
            raw_scores_per_stride.append(np.reshape(raw_scores.numpy(),
                                                    [-1, model.num_frames // 2]))
            within_period_score_stride.append(np.reshape(within_period_scores.numpy(),
                                                         [-1, 1]))
            embs_stride.append(embs)
            if progress_cb:
                progress_cb((100 * Fprg * float(i * num_batches + batch_idx + 1)) / Nprg)
        raw_scores_per_stride = np.concatenate(raw_scores_per_stride, axis=0)
        raw_scores_list.append(raw_scores_per_stride)
        within_period_score_stride = np.concatenate(
            within_period_score_stride, axis=0)
        pred_score, within_period_score_stride = get_score(
            raw_scores_per_stride, within_period_score_stride)
        scores.append(pred_score)
        within_period_scores_list.append(within_period_score_stride)
        embs_list.append(np.concatenate(embs_stride, axis=0))

    # Stride chooser
    argmax_strides = np.argmax(scores)
    chosen_stride = strides[argmax_strides]
    raw_scores = np.repeat(
        raw_scores_list[argmax_strides], chosen_stride, axis=0)[:seq_len]

    final_embs = embs_list[argmax_strides]

    # QRS
    within_period_scores = within_period_scores_list[argmax_strides]
    feat_factors = []
    if pcaks:
        start_time = time.time()
        factors = np.ones(len(final_embs))
        scaler = pcaks['scaler']
        pca = pcaks['pca']
        ecdfs = pcaks['ecdfs']
        alpha = pcaks.get('alpha', 0.01)
        beta = pcaks.get('beta', 0.5)
        gamma = pcaks.get('gamma', 0.7)
        ks_thresh = beta * sum(pca.explained_variance_ratio_)
        embs_feat = np.concatenate(final_embs, axis=0)
        pca_out = pca.transform(scaler.transform(embs_feat))
        tfp_cdf = ecdfs.cdf(pca_out).numpy()
        print(embs_feat.shape, pca_out.shape, tfp_cdf.shape) 
        M = len(final_embs)
        for i, j in enumerate(range(0, pca_out.shape[0], N)):
            indices = np.argsort(pca_out[j:j + N], axis=0)
            cdfvals = np.take_along_axis(tfp_cdf[j:j + N], indices, axis=0)
            # Dmin = (cdfvals - CDF0).max(axis=0)
            # Dplus = (CDF1 - cdfvals).max(axis=0)
            # D = np.max([Dmin, Dplus], axis=0)
            D = np.abs(CDF1 - cdfvals).max(axis=0)
            pvals = []
            for d in D:
                pvalue = 2 * stats.distributions.ksone.sf(d, N)
                pvals.append(pvalue)
            pvals = np.array(pvals, dtype=np.float)
            pvalue = np.clip(pvals, 0, 1)
            ksret = sum(pca.explained_variance_ratio_[pvals > alpha])
            if ksret < ks_thresh:
                factors[i] = round(gamma * ksret / ks_thresh, 2)
            feat_factors.append((ksret, factors[i]))
            if progress_cb:
                progress_cb(100 * Fprg * (1 + float(i + 1) / M))
        within_period_scores *= factors.repeat(64)
        print('pcakstest time: %d secs' % (time.time() - start_time))

    within_period = np.repeat(
        within_period_scores, chosen_stride,
        axis=0)[:seq_len]
    within_period_binary = np.asarray(within_period > within_period_threshold)
    if median_filter:
        within_period_binary = medfilt(within_period_binary, 5)

    if constant_speed:
        # Select Periodic frames
        periodic_idxes = np.where(within_period_binary)[0]

        # Count by averaging predictions. Smoother but
        # assumes constant speed.
        scores = tf.reduce_mean(
            tf.nn.softmax(raw_scores[periodic_idxes], axis=-1), axis=0)
        max_period = np.argmax(scores)
        pred_score = scores[max_period]
        pred_period = chosen_stride * (max_period + 1)
        per_frame_counts = (
                np.asarray(seq_len * [1. / pred_period]) * # noqa
                np.asarray(within_period_binary))
    else:
        # Count each frame. More noisy but adapts to changes in speed.
        pred_score = tf.reduce_mean(within_period)
        per_frame_periods = tf.argmax(raw_scores, axis=-1) + 1
        per_frame_counts = tf.where(
            tf.math.less(per_frame_periods, 3),
            0.0,
            tf.math.divide(1.0,
                           tf.cast(chosen_stride * per_frame_periods, tf.float32)),
        )
        if median_filter:
            per_frame_counts = medfilt(per_frame_counts, 5)

        per_frame_counts *= np.asarray(within_period_binary)

    # QRS
    cnts = np.sum(per_frame_counts)
    if cnts > 0:
        pred_period = seq_len / np.sum(per_frame_counts)
    else:
        pred_period = seq_len * 0.0

    # feature map
    if osd_feat:
        idxes = tf.range(0, len(frames), chosen_stride)
        chosen_frames = tf.gather(frames, idxes)
        feature_maps = model.base_model.predict(chosen_frames)
    else:
        feature_maps = None

    del frames, scores, within_period_scores_list, embs_list

    if pred_score < threshold:
        print('No repetitions detected in video as score '
              '%0.2f is less than threshold %0.2f.' % (pred_score, threshold))
        per_frame_counts = np.asarray(len(per_frame_counts) * [0.])

    return (pred_period, pred_score, within_period,
            per_frame_counts, chosen_stride, final_embs, feature_maps, feat_factors)


def get_score(period_score, within_period_score):
    """Combine the period and periodicity scores."""
    within_period_score = tf.nn.sigmoid(within_period_score)[:, 0]
    per_frame_periods = tf.argmax(period_score, axis=-1) + 1
    pred_period_conf = tf.reduce_max(
        tf.nn.softmax(period_score, axis=-1), axis=-1)
    pred_period_conf = tf.where(
        tf.math.less(per_frame_periods, 3), 0.0, pred_period_conf)
    within_period_score *= pred_period_conf
    within_period_score = np.sqrt(within_period_score)
    pred_score = tf.reduce_mean(within_period_score)
    return pred_score, within_period_score
