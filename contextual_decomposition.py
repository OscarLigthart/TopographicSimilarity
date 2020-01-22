from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.attention import CDAttention
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.corpus import Corpus
from diagnnose.vocab import get_vocab_from_config
from diagnnose.extractors.base_extractor import Extractor

arg_groups = {
    "activations",
    "corpus",
    "decompose",
    "extract",
    "init_states",
    "model",
    "plot_attention",
    "vocab",
}
arg_parser, required_args = create_arg_parser(arg_groups)

print(arg_parser)
print(required_args)

config_dict = create_config_dict(arg_parser, required_args, arg_groups)

print(config_dict)