from pathlib import Path

from LieConv.lie_conv.lieGroups import SE3

from point_vs.models.egnn_lucid import PygLucidEGNN
from point_vs.models.egnn_satorras import SartorrasEGNN
from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from point_vs.utils import load_yaml


def load_model(weights_file):
    model_path = Path(weights_file).expanduser()
    model_kwargs = load_yaml(model_path.parents[1] / 'model_kwargs.yaml')
    model_kwargs['group'] = SE3(0.2)
    cmd_line_args = load_yaml(model_path.parents[1] / 'cmd_args.yaml')
    model_type = cmd_line_args['model']

    model_class = {
        'lieconv': LieResNet,
        'egnn': SartorrasEGNN,
        'lucid': PygLucidEGNN,
        'lietransformer': EquivariantTransformer
    }

    model_class = model_class[model_type]
    model = model_class(Path(), learning_rate=0, weight_decay=0,
                        silent=True, **model_kwargs)

    model.load_weights(model_path)
    model = model.eval()

    return model, model_kwargs, cmd_line_args