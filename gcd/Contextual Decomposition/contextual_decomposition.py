from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.attention import CDAttention
from diagnnose.models.import_model import import_model
from diagnnose.typedefs.models import LanguageModel
from diagnnose.typedefs.corpus import Corpus
from diagnnose.vocab import get_vocab_from_config


if __name__ == "__main__":
    arg_groups = {
        "activations",
        "corpus",
        "decompose",
        "init_states",
        "model",
        "plot_attention",
        "vocab",
        "extract"
    }

    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args)

    # load the model
    model: LanguageModel = import_model(config_dict)

    # load the corpus
    corpus: Corpus = import_corpus(
        vocab_path=get_vocab_from_config(config_dict), **config_dict["corpus"]
    )

    # set fix shapley to false
    if "fix_shapley" not in config_dict["decompose"]:
        config_dict["decompose"]["fix_shapley"] = False

    # get attention?
    attention = CDAttention(
        model,
        corpus,
        cd_config=config_dict["decompose"],
        plot_config=config_dict["plot_attention"],
    )

    print("Creating example plot")
    attention.plot_by_sen_id(
        [2], avg_decs=True
    )
