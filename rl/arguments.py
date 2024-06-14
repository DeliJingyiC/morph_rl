import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    parser.add_argument("--language")
    parser.add_argument("--run")
    parser.add_argument("--force_test")
    parser.add_argument("--pos_target", default="VERB")
    parser.add_argument("--ud_train", nargs="+")
    parser.add_argument("--unimorph_data")
    parser.add_argument("--split")
    parser.add_argument("--local_frequency", help="Use empirical log freq instead of Subtlex", action="store_true")
    parser.add_argument("--ud_dataframe")
    parser.add_argument("--transformer_output")
    parser.add_argument("--single_source", action="store_true")
    parser.add_argument("--noncumulative", action="store_true")
    parser.add_argument("--synthetic_multitask", action="store_true")
    parser.add_argument("--multitask_only", action="store_true")
    parser.add_argument("--char_encode_features", action="store_true")
    parser.add_argument("--epoch", default=0, type=int)
    parser.add_argument("--cache_section", type=int)
    args = parser.parse_args()
    return args
