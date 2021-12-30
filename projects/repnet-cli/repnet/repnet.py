import numpy as np
import time
import tensorflow as tf
from scipy.signal import medfilt
from repnet import ResnetPeriodEstimator

from scipy import stats


N = 64

CDF0 = (np.arange(0, N) / N).reshape((-1, 1))
CDF1 = (np.arange(1.0, N + 1) / N).reshape((-1, 1))


def empirical_kstest(emb, scaler, pca, ecdfs, alpha=0.01):
    out = pca.transform(scaler.transform(emb))
    # pvals = np.array([stats.kstest(out[:, i], cdf=lambda x: ecdfs[i](x))[1] for i in range(out.shape[-1])])
    pvals = np.array([stats.kstest(out[:, i], cdf=lambda x: ecdfs[i](x))[1] for i in range(out.shape[-1])])
    return sum(pca.explained_variance_ratio_[pvals > alpha])


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


def create_count_video(frames,
                       per_frame_counts,
                       within_period,
                       score,
                       fps,
                       output_file,
                       delay,
                       plot_count=True,
                       plot_within_period=False,
                       plot_score=False,
                       vizualize_reps=False,
                       progress_cb=None):
    """Creates video with running count and within period predictions.

  Args:
    frames (List): List of images in form of NumPy arrays.
    per_frame_counts (List): List of floats indicating repetition count for
      each frame. This is the rate of repetition for that particular frame.
      Summing this list up gives count over entire video.
    within_period (List): List of floats indicating score between 0 and 1 if the
      frame is inside the periodic/repeating portion of a video or not.
    score (float): Score between 0 and 1 indicating the confidence of the
      RepNet model's count predictions.
    fps (int): Frames per second of the input video. Used to scale the
      repetition rate predictions to Hz.
    output_file (string): Path of the output video.
    delay (integer): Delay between each frame in the output video.
    plot_count (boolean): if True plots the count in the output video.
    plot_within_period (boolean): if True plots the per-frame within period
      scores.
    plot_score (boolean): if True plots the confidence of the model along with
      count ot within_period scores.
  """
    if output_file[-4:] not in ['.mp4', '.gif']:
        raise ValueError('Output format can only be mp4 or gif')

    if vizualize_reps:
        return viz_reps(frames, per_frame_counts, score, output_file,
                interval=delay, plot_score=plot_score,
                progress_cb=progress_cb)

    num_frames = len(frames)

    running_counts = np.cumsum(per_frame_counts)
    final_count = np.around(running_counts[-1]).astype(np.int)

    def count(idx):
        return int(np.round(running_counts[idx]))

    def rate(idx):
        return per_frame_counts[idx] * fps

    if plot_count and not plot_within_period:
        fig = plt.figure(figsize=(10, 12), tight_layout=True)
        im = plt.imshow(unnorm(frames[0]))
        if plot_score:
            plt.suptitle('Pred Count: %d, '
                         'Prob: %0.1f' % (final_count, score),
                         fontsize=24)

        plt.title('Count 0, Rate: 0', fontsize=24)
        plt.axis('off')
        plt.grid(b=None)

        def update_count_plot(i):
            """Updates the count plot."""
            im.set_data(unnorm(frames[i]))
            plt.title('Count %d, Rate: %0.4f Hz' % (count(i), rate(i)), fontsize=24)

        anim = FuncAnimation(
            fig,
            update_count_plot,
            frames=np.arange(1, num_frames),
            interval=delay,
            blit=False)
        if output_file[-3:] == 'mp4':
            anim.save(f"{output_file[:-4]}_{final_count:02d}.mp4", dpi=100, fps=None)
        elif output_file[-3:] == 'gif':
            anim.save(f"{output_file[:-4]}_{final_count:02d}.gif", writer='imagemagick', fps=None, dpi=100)

    elif plot_within_period:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        im = axs[0].imshow(unnorm(frames[0]))
        axs[1].plot(0, within_period[0])
        axs[1].set_xlim((0, len(frames)))
        axs[1].set_ylim((0, 1))

        if plot_score:
            plt.suptitle('Pred Count: %d, '
                         'Prob: %0.1f' % (final_count, score),
                         fontsize=24)

        if plot_count:
            axs[0].set_title('Count 0, Rate: 0', fontsize=20)

        plt.axis('off')
        plt.grid(b=None)

        def update_within_period_plot(i):
            """Updates the within period plot along with count."""
            im.set_data(unnorm(frames[i]))
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            xs = []
            ys = []
            if plot_count:
                axs[0].set_title('Count %d, Rate: %0.4f Hz' % (count(i), rate(i)),
                                 fontsize=20)
            for idx in range(i):
                xs.append(idx)
                ys.append(within_period[int(idx * len(within_period) / num_frames)])
            axs[1].clear()
            axs[1].set_title('Within Period or Not', fontsize=20)
            axs[1].set_xlim((0, num_frames))
            axs[1].set_ylim((-0.05, 1.05))
            axs[1].plot(xs, ys)

        anim = FuncAnimation(
            fig,
            update_within_period_plot,
            frames=np.arange(1, num_frames),
            interval=delay,
            blit=False,
        )

        if output_file[-3:] == 'mp4':
            anim.save(output_file, dpi=100, fps=None)
        elif output_file[-3:] == 'gif':
            anim.save(output_file, writer='imagemagick', fps=None, dpi=100)

    plt.close()


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
        # for i in range(len(final_embs)):
        #     emb = final_embs[i]
        #     ksret = empirical_kstest(emb, scaler, pca, ecdfs, alpha)
        #     if ksret < ks_thresh:
        #         factors[i] = round(gamma * ksret / ks_thresh, 2)
        #     feat_factors.append((ksret, factors[i]))
        #     if progress_cb:
        #         progress_cb(100 * Fprg * (1 + float(i + 1) / len(final_embs)))
        embs_feat = np.concatenate(final_embs, axis=0)
        pca_out = pca.transform(scaler.transform(embs_feat))
        tfp_cdf = ecdfs.cdf(pca_out).numpy()
        print(embs_feat.shape, pca_out.shape, tfp_cdf.shape) 
        M = len(final_embs)
        for i in range(M):
            indices = np.argsort(pca_out[i:i + N], axis=0)
            cdfvals = np.take_along_axis(tfp_cdf[i:i + N], indices, axis=0)
            # D = np.abs(CDF1 - cdfvals).max(axis=0)
            Dmin = (cdfvals - CDF0).max(axis=0)
            Dplus = (CDF1 - cdfvals).max(axis=0)
            D = np.max([Dmin, Dplus], axis=0)
            pvals = []
            for d in D:
                pvalue = 2 * stats.distributions.ksone.sf(d, N)
                pvalue = np.clip(pvalue, 0, 1)
                pvals.append(pvalue)
            print(i, pvals)
            pvals = np.array(pvals, dtype=np.float)
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
                np.asarray(seq_len * [1. / pred_period]) *
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
