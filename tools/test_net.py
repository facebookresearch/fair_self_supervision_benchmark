# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import logging
import numpy as np
import os
import sys

from self_supervision_benchmark.core.config import config as cfg
from self_supervision_benchmark.core.config import (
    cfg_from_file, cfg_from_list, assert_cfg, print_cfg
)
import self_supervision_benchmark.metrics.metrics_topk as metrics_topk
import self_supervision_benchmark.metrics.metrics_ap as metrics_ap
from self_supervision_benchmark.utils import helpers, checkpoints
from self_supervision_benchmark.utils.timer import Timer
from self_supervision_benchmark.modeling import model_builder

from caffe2.python import workspace

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# import the custom ops provided by detectron
helpers.import_detectron_ops()


def test_net(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)

    prefix, device = helpers.get_prefix_and_device()

    ############################################################################
    name = '{}_test'.format(cfg.MODEL.MODEL_NAME)
    logger.info('=================Creating model: {}============='.format(name))
    data_type, batch_size = cfg['TEST'].DATA_TYPE, cfg['TEST'].BATCH_SIZE
    test_model = model_builder.ModelBuilder(
        name=name, train=False, use_cudnn=True, cudnn_exhaustive_search=True,
        split=data_type, ws_nbytes_limit=(cfg.CUDNN_WORKSPACE_LIMIT * 1024 * 1024),
    )

    test_model.build_model()
    test_model.create_net()
    test_model.start_data_loader()

    assert cfg.METRICS.TYPE in ['topk', 'AP'], "Invalid metrics type"
    test_metrics_calculator = None
    if cfg.METRICS.TYPE == 'topk':
        test_metrics_calculator = metrics_topk.TopkMetricsCalculator(
            model=test_model, split=data_type, batch_size=batch_size, prefix=prefix
        )
    else:
        test_metrics_calculator = metrics_ap.APMetricsCalculator(
            model=test_model, split=data_type, batch_size=batch_size, prefix=prefix
        )

    test_timer = Timer()
    total_test_iters = helpers.get_num_test_iter(test_model.input_db)
    logger.info('Test epoch iters: {}'.format(total_test_iters))

    # save proto for debugging
    helpers.save_model_proto(test_model)

    ############################################################################
    # initialize the model from the checkpoint
    if cfg.TEST.PARAMS_FILE:
        checkpoints.load_model_from_params_file(
            test_model, params_file=cfg.TEST.PARAMS_FILE, checkpoint_dir=None)
    else:
        logger.info('No params files specified for testing model. Aborting!')
        os._exit(0)

    ############################################################################
    logger.info("Testing model...")
    test_metrics_calculator.reset()
    for test_iter in range(0, total_test_iters):
        test_timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        test_timer.toc()
        if test_iter == 0:
            helpers.print_net(test_model)
        rem_test_iters = total_test_iters - test_iter - 1
        test_metrics_calculator.calculate_and_log_test_iter_metrics(
            test_iter, test_timer, rem_test_iters, total_test_iters
        )
    test_metrics_calculator.finalize_metrics()
    test_metrics_calculator.compute_and_log_epoch_best_metric(model_iter=test_iter)
    test_metrics_calculator.log_best_model_metrics(test_iter, total_test_iters)
    logger.info('Total images tested: {}'.format(test_metrics_calculator.split_N))
    logger.info('Done!!!')
    test_model.data_loader.shutdown_dataloader()

def vis(opts):
    import matplotlib.pyplot as plt
    from tempfile import TemporaryFile
    outdir = '/home/yihuihe/outs/'
    pathname = os.path.join(outdir, 'check_' + os.path.splitext(os.path.basename(opts.config_file))[0])
    try:
        os.makedirs(pathname)
    except:
        # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python
        for the_file in os.listdir(pathname):
            file_path = os.path.join(pathname, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)
    output_dir = helpers.get_output_directory()
    test_model = model_builder.ModelBuilder(
        name='{}_test'.format(cfg.MODEL.MODEL_NAME), train=False,
        use_cudnn=True, cudnn_exhaustive_search=True,
        split=cfg.TEST.DATA_TYPE
    )
    test_model.build_model() #output_dir=output_dir)
    test_model.create_net()
    total_test_net_iters = helpers.get_num_test_iter(test_model.input_db)
    test_model.start_data_loader()
    prefix = 'gpu_{}'.format(test_model.GetDevices()[0])
    checkpoints.load_model_from_params_file(
        test_model, params_file=cfg.TRAIN.PARAMS_FILE, checkpoint_dir=None)
    timer = Timer()
    idx = 0
    def label2img(lbl):
        return [plt.imread(os.path.join(outdir, 'vis' + str(cfg.MODEL.NUM_CLASSES), str(int(l)) + '.png')) for l in lbl]

    for _test_iter in range(0, total_test_net_iters):
        print(idx)
        timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        data = []
        data.append(workspace.FetchBlob('gpu_0/data_0'))
        data.append(workspace.FetchBlob('gpu_0/data_1'))
        data.append(workspace.FetchBlob('gpu_0/data_2'))
        data.append(workspace.FetchBlob('gpu_0/data_3'))
        data.append(workspace.FetchBlob('gpu_0/data_4'))
        data.append(workspace.FetchBlob('gpu_0/data_5'))
        data.append(workspace.FetchBlob('gpu_0/data_6'))
        data.append(workspace.FetchBlob('gpu_0/data_7'))
        data.append(workspace.FetchBlob('gpu_0/data_8'))
        # fullimgs = workspace.FetchBlob('gpu_0/data')
        # fullimgs = deprocess_imgs(fullimgs)
        # N x K
        labels = workspace.FetchBlob('gpu_0/_labels')
        labels_prob = labels.copy()
        labels = (-labels.squeeze()).argsort()
        individual_labels = workspace.FetchBlob('gpu_0/individual_labels').squeeze()
        individual_labels_prob = individual_labels.copy()
        individual_labels = (-individual_labels).argsort()
        preds = workspace.FetchBlob('gpu_0/softmax').squeeze()
        preds_prob = preds.copy()
        preds = (-preds).argsort()
        imgs = []
        for i in range(len(data[0])):
            img = []
            for j in data:
                img.append(j[i])
            img = deprocess_imgs(np.array(img))
            # print(img.min(), img.max())
            tmp = 'tmp.png' #TemporaryFile()
            show_images(img[[0, 3, 6, 1, 4, 7, 2, 5, 8]], BGR=False,
                name=tmp)
            imgs.append(plt.imread(tmp))
        # for fullimg, img, label in zip(fullimgs, imgs, labels):
        for n, (img, label, label_prob, pred, pred_prob) in enumerate(zip(imgs, labels, labels_prob, preds, preds_prob)):
            topk = 8
            print(label[:5])
            toshow, titles, pos = [], [], []

            toshow += [img]
            titles += ['input']
            pos += [1]
            label_imgs = label2img(label[:topk])
            toshow += label_imgs
            titles += ['gt(avg)_top{}_{:.2f}'.format(i+1, float(label_prob[label[i]])) for i in range(topk)]
            pos += [topk + 1 + i for i in range(topk)]

            individual_label = individual_labels[:, n, 0]
            individual_label_prob = [individual_labels_prob[i, n, individual_label[i]] for i in range(8)]
            individual_labels_imgs = label2img(individual_label)
            toshow += [individual_labels_imgs[i] for i in [0,3,5,1, 6,2,4,7]] # [0,3,6,1, 7,2,5,8]]
            titles += ['p{}_top1_{:.2f}'.format(i, float(individual_label_prob[k])) for k, i in enumerate([0,1,2,3, 5,6,7,8])]
            pos += [2 * topk + 1 + i for i in range(topk)]

            pred_imgs = label2img(pred[:topk])
            toshow += pred_imgs
            titles += ['pred_top{}_{:.2f}'.format(i+1, float(pred_prob[pred[i]])) for i in range(topk)]
            pos += [3 * topk + 1 + i for i in range(topk)]

            for i in range(len(titles)):
                titles[i] = titles[i].replace('0.', '.')


            # show_images([fullimg] + [img]+label_imgs,
            show_images(toshow,
                cols=4,
                titles=titles,
                name=os.path.join(pathname, str(idx)),
                BGR=False,
                pos=pos,
                size=(8, 4))
            idx += 1

        timer.toc()

    test_model.data_loader.shutdown_dataloader()

def show_images(images, cols=None, titles=None, name='tmp', BGR=True, pos=None, size=None):
    """https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    import matplotlib.pyplot as plt
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if cols is None:
        cols = int(n_images**.5)

    if pos is None:
        fig = plt.figure(figsize=(np.ceil(n_images/float(cols)), cols))
    else:
        assert size is not None
        fig = plt.figure(figsize=size)

    for n, image in enumerate(images):
        if pos is None:
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        else:
            a = fig.add_subplot(size[1], size[0], pos[n])

        a.set_axis_off()
        if image.ndim == 2:
            plt.gray()
        if BGR:
            image = image[..., ::-1]
        plt.imshow(image, aspect="auto")
        if titles is not None:
            a.set_title(titles[n], fontsize=5, y=0.9)
    #fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.subplots_adjust(wspace=.05, hspace=.05, )#left=0, bottom=0, right=1, top=1)
    if isinstance(name, str) and '.png' not in name:
        name += '.png'
    plt.savefig(name, dpi=300)
    plt.close('all')

def deprocess_imgs(imgs):
    DATA_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    DATA_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    imgs = imgs * DATA_STD[:, None, None] + DATA_MEAN[:, None, None]
    imgs = imgs.transpose((0, 2, 3, 1)) # * 255
    return imgs[..., ::-1]

def main():
    parser = argparse.ArgumentParser(description='Model testing')
    parser.add_argument('--action', type=str, default='test_net',
                        help='test_net | vis')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
    parser.add_argument('--args_dict', type=str, default=None,
                        help='Args can also be passed as a dict.')
    parser.add_argument('opts', help='see configs.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_cfg()
    print_cfg()
    # test_net(args)
    globals()[args.action](args)

if __name__ == '__main__':
    main()
