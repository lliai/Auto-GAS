import unittest

# Import the module to be tested
from gaswot.api import TransNASBenchAPI


class TestTransNASBenchAPI(unittest.TestCase):

    def test_transnasbenchapi(self):
        # Test the TransNASBenchAPI class
        api = TransNASBenchAPI('./data/transnas-bench_v10141024.pth')
        import pdb
        pdb.set_trace()
        # xarch = '64-4111-basic'
        xarch = '64-41414-3_33_212'
        for xtask in api.task_list:
            print(f'----- {xtask} -----')
            print('--- info ---')
            for xinfo in api.info_names:
                print(f'{xinfo} : {api.get_model_info(xarch, xtask, xinfo)}')
            print('--- metrics ---')
            for xmetric in api.metrics_dict[xtask]:
                print(
                    f"{xmetric} : {api.get_single_metric(xarch, xtask, xmetric, mode='best')}"
                )
                print(
                    f'best epoch : {api.get_best_epoch_status(xarch, xtask, metric=xmetric)}'
                )
                print(
                    f'final epoch : {api.get_epoch_status(xarch, xtask, epoch=-1)}'
                )
                if ('valid' in xmetric and 'loss' not in xmetric) or (
                        'valid' in xmetric and 'neg_loss' in xmetric):
                    print(
                        f"\nbest_arch -- {xmetric}: {api.get_best_archs(xtask, xmetric, 'micro')[0]}"
                    )


if __name__ == '__main__':
    unittest.main()

    metrics_dict = {
        'class_scene': [
            'train_top1', 'train_top5', 'train_loss', 'valid_top1',
            'valid_top5', 'valid_loss', 'test_top1', 'test_top5', 'test_loss',
            'time_elapsed'
        ],
        'class_object': [
            'train_top1', 'train_top5', 'train_loss', 'valid_top1',
            'valid_top5', 'valid_loss', 'test_top1', 'test_top5', 'test_loss',
            'time_elapsed'
        ],
        'room_layout': [
            'train_loss', 'train_neg_loss', 'valid_loss', 'valid_neg_loss',
            'test_loss', 'test_neg_loss', 'time_elapsed'
        ],
        'jigsaw': [
            'train_top1', 'train_top5', 'train_loss', 'valid_top1',
            'valid_top5', 'valid_loss', 'test_top1', 'test_top5', 'test_loss',
            'time_elapsed'
        ],
        'segmentsemantic': [
            'train_loss', 'train_acc', 'train_mIoU', 'valid_loss', 'valid_acc',
            'valid_mIoU', 'test_loss', 'test_acc', 'test_mIoU', 'time_elapsed'
        ],
        'normal': [
            'train_ssim', 'train_l1_loss', 'valid_ssim', 'valid_l1_loss',
            'test_ssim', 'test_l1_loss', 'time_elapsed'
        ],
        'autoencoder': [
            'train_ssim', 'train_l1_loss', 'valid_ssim', 'valid_l1_loss',
            'test_ssim', 'test_l1_loss', 'time_elapsed'
        ]
    }

    info_names = [
        'inference_time', 'encoder_params', 'model_params', 'model_FLOPs',
        'encoder_FLOPs'
    ]

    database_task_list = [
        'class_scene', 'class_object', 'room_layout', 'jigsaw',
        'segmentsemantic', 'normal', 'autoencoder'
    ]

    database_data_macro = [
        '64-43243-basic', '64-41443-basic', '64-23443-basic', '64-43432-basic',
        '64-43414-basic', '64-43234-basic', '64-41434-basic', '64-23434-basic',
        '64-43342-basic', '64-43324-basic', '64-43144-basic', '64-41344-basic',
        '64-23344-basic'
    ]

    database_data_micro = [
        '64-41414-3_33_212', '64-41414-3_33_213', '64-41414-3_33_220',
        '64-41414-3_33_221', '64-41414-3_33_222', '64-41414-3_33_223',
        '64-41414-3_33_230', '64-41414-3_33_231', '64-41414-3_33_232',
        '64-41414-3_33_233', '64-41414-3_33_300', '64-41414-3_33_301',
        '64-41414-3_33_302', '64-41414-3_33_303', '64-41414-3_33_310'
    ]

    single_arch_keys = [
        'class_scene', 'class_object', 'room_layout', 'jigsaw',
        'segmentsemantic', 'normal', 'autoencoder'
    ]
