import config
from optparse import OptionParser
from task import Task
from utils import logging_utils
from model_param_space import param_space_dict
import datetime

def parse_args(parser):
    parser.add_option("-m", "--model", dest="model_name", type="string")
    parser.add_option("-e", "--eval", dest="eval", default=False, action="store_true")
    parser.add_option("-r", "--runs", dest="runs", default=3, type="int")
    parser.add_option("-p", "--prefix", dest="prefix", type="string", default="test")

    options, args = parser.parse_args()
    return options, args

def main(options):
    if options.eval:
        time_str = datetime.datetime.now().isoformat()
        logname = "Eval_[Model@%s]_%s.log" % (options.model_name, time_str)
        logger = logging_utils._get_logger(config.LOG_DIR, logname)
    else:
        time_str = datetime.datetime.now().isoformat()
        logname = "Final_[Model@%s]_%s.log" % (options.model_name, time_str)
        logger = logging_utils._get_logger(config.LOG_DIR, logname)
    params_dict = param_space_dict[options.model_name]
    task = Task(options.model_name, options.runs, params_dict, logger)
    if options.eval:
        task.refit(options.prefix)
    else:
        task.evaluate(options.prefix)

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
