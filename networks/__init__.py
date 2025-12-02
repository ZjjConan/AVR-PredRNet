from .resnet4b import resnet4b
from .predrnet import predrnet_raven, predrnet_vad
from .sspredrnet import sspredrnet_raven, sspredrnet_vad

model_dict = {
    "resnet4b": resnet4b,
    "predrnet_raven": predrnet_raven,
    "predrnet_vad": predrnet_vad,
    "sspredrnet_raven": sspredrnet_raven,
    "sspredrnet_vad": sspredrnet_vad
}


def create_net(args):
    net = None

    kwargs = {}
    kwargs["block_drop"] = args.block_drop
    kwargs["classifier_drop"] = args.classifier_drop
    kwargs["classifier_hidreduce"] = args.classifier_hidreduce
    kwargs["num_filters"] = args.num_filters
    kwargs["num_extra_stages"] = args.num_extra_stages
    kwargs["in_channels"] = args.in_channels
    kwargs["enable_rc"] = args.enable_rc

    net = model_dict[args.arch.lower()](**kwargs)

    return net

