hparams = {
    'lr': 5e-5,
    'batch_size': 64,
    'student': {
        'pretrained': 'bert-base-uncased',
        'pretrained_dim': 768,
        'num_aspect': 9,
    },
    'description': 'bag baseline',
    'save_dir': './ckpt/bags',
    'aspect_init_file': './data/seedwords/bags_and_cases.5.txt',
    'train_file': './data/bags_train.json',
    'test_file': './data/bags_test.json',
    'general_asp': 5,
    'maxlen': 40
}
