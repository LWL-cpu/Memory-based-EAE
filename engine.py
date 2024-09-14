import os
import logging
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from models import build_model
from processors import build_processor
from utils import set_seed
from runner.runner import Runner

logger = logging.getLogger(__name__)


def run(args, model, processor, optimizer, scheduler):
    set_seed(args)

    logger.info("train dataloader generation")
    train_examples, train_features, train_dataloader, args.train_invalid_num = processor.generate_dataloader('train')
    logger.info("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = processor.generate_dataloader('dev')
    logger.info("test dataloader generation")
    test_examples, test_features, test_dataloader, args.test_invalid_num = processor.generate_dataloader('test')

    runner = Runner(
        cfg=args,
        data_samples=[train_examples, dev_examples, test_examples],
        data_features=[train_features, dev_features, test_features],
        data_loaders=[train_dataloader, dev_dataloader, test_dataloader],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn_dict=None,
    )
    out = runner.run()
    return out


def main(space):
    from config_parser import get_args_parser
    args = get_args_parser()

    if not args.inference_only:
        print(f"Output full path {os.path.join(os.getcwd(), args.output_dir)}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logging.basicConfig(
            filename=os.path.join(args.output_dir, "log.txt"), \
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt='%m/%d/%Y %H:%M:%S', level = logging.INFO
            )
    else:
        logging.basicConfig(
            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
            )

    args.seed = int(space[0][1])
    args.infer_batch_size = space[1][1]
    args.memory_layers = space[2][1]
    args.lr_gate = space[3][1]
    print("seed:", args.seed, "infer_batch_size:", args.infer_batch_size, "memory_layers:",
          args.memory_layers, "lr_gate:", args.lr_gate)
    set_seed(args)

    model, tokenizer, optimizer, scheduler = build_model(args, args.model_type) 
    model.to(args.device)

    processor = build_processor(args, tokenizer)

    logger.info("Training/evaluation parameters %s", args)
    out = run(args, model, processor, optimizer, scheduler)
    return out
            

if __name__ == "__main__":
    space = [
        ('seed', hp.choice('seed', [666, 999, 3417, 42, 111, 66, 1111])),
        ('infer_batch_size', hp.choice('infer_batch_size', [1])),
        ('memory_layers', hp.choice('memory_layers', [6, 8, 10, 12])),
        ('lr_gate', hp.choice('lr_gate', [ 1e-4, 5e-4, 1e-5, 3e-5]))]

    # minimize the objective over the space

    best = fmin(main, space, algo=tpe.suggest, max_evals=20)

    print(best)
    # -> {'a': 1, 'c2': 0.01420615366247227}
    print(space_eval(space, best))
