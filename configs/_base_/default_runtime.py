checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl', port=29509)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
