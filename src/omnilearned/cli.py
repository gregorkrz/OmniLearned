import typer
import json
import os
from types import SimpleNamespace
from omnilearned.train import run as run_training
from omnilearned.evaluate import run as run_evaluation
from omnilearned.dataloader import load_data, Task
from omnilearned.train_hl import run as run_training_hl
from omnilearned.evaluate_hl import run as run_evaluation_hl

"""
# Eval 29 February 2026

export CKPT_DIR=/global/cfs/cdirs/m3246/gregork/checkpoints
omnilearned evaluate -i $CKPT_DIR/current_type_1A_20260217_175547
omnilearned evaluate -i $CKPT_DIR/current_type_1A_20260218_015241
omnilearned evaluate -i $CKPT_DIR/pions_1A_20260218_014607
omnilearned evaluate -i $CKPT_DIR/pions_1A_20260218_014329
omnilearned evaluate -i $CKPT_DIR/E_avail_no_muon_1A_20260218_064743 --num-workers 2 --batch 100
omnilearned evaluate -i $CKPT_DIR/E_avail_1A_20260218_064634 --num-workers 2 --batch 100
omnilearned evaluate -i $CKPT_DIR/E_avail_1A_20260218_064356 --num-workers 2 --batch 100
omnilearned evaluate -i $CKPT_DIR/E_avail_no_muon_1A_20260218_064440 --num-workers 2 --batch 100

omnilearned evaluate -i $CKPT_DIR/E_avail_no_muon_Linear_Huber2_1A_20260223_193110 --num-workers 2 --batch 100
omnilearned evaluate -i $CKPT_DIR/E_avail_no_muon_Linear_Huber2_1A_20260223_193233 --num-workers 2 --batch 100


omnilearned evaluate -i $CKPT_DIR/Train_CC1orNPi_Fix250226_PTCont_1A_20260226_033635 --num-workers 2 --batch 1024
omnilearned evaluate -i $CKPT_DIR/Train_CC1orNPi_Fix250226Cont_1A_20260226_033607 --num-workers 2 --batch 1024
"""

app = typer.Typer(
    help="OmniLearned: A unified deep learning approach for particle physics",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
# export CKPT_DIR=/global/cfs/cdirs/m3246/gregork/checkpoints

def create_task(args):
    if args.mode == "regression":
        class_label_idx = None
        class_idx = None
        if args.regress_E_available:
            class_label_idx = 8
        elif args.regress_E_available_no_muon:
            class_label_idx = 9
        return Task(type="regression", regress_E_available=args.regress_E_available, regress_E_available_no_muon=args.regress_E_available_no_muon,
                class_label_idx=class_label_idx, regress_log=args.regress_log)
    elif args.mode == "classifier":
        if "classification_n_pions" not in args.__dict__:
            args.classification_n_pions = False
        if args.classification_event_type:
            class_label_idx = 1
            class_idx = [1, 2, 3, 4, 8]
            class_idx_map = {1: 0, 2: 1, 3: 2, 4: 3, 8: 4}
        elif args.classification_current:
            class_label_idx = 3
            class_idx = [1, 2]
            class_idx_map = {1: 0, 2: 1}
        elif args.classification_cc_1pi:
            class_label_idx = 4
            class_idx = [0, 1, 2]
            class_idx_map = {0: 0, 1: 1, 2: 2}
        elif args.classification_n_pions:
            class_label_idx = 7
            class_idx = [0, 1]
            class_idx_map = {0: 0, 1: 1}
        elif args.classification_CC1orNPi:
            class_label_idx = -1
            class_idx = [0, 1, 2, 3, 4]
            class_idx_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        return Task(type="classifier", classification_event_type=args.classification_event_type, 
            classification_current=args.classification_current, classification_cc_1pi=args.classification_cc_1pi,
            classification_n_pions=args.classification_n_pions, class_idx=class_idx, class_idx_map=class_idx_map, class_label_idx=class_label_idx,
            classification_CC1orNPi=args.classification_CC1orNPi)
    else:
        raise ValueError("Invalid mode")


@app.command()
def train(
    # General Options
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output best model"
    ),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    pretrain_tag: str = typer.Option(
        "", help="Tag given to pretrained checkpoint model"
    ),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    wandb: bool = typer.Option(False, help="use wandb logging"),
    fine_tune: bool = typer.Option(False, help="Fine tune the model"),
    resuming: bool = typer.Option(False, help="Resume training"),
    # Model Options
    num_feat: int = typer.Option(
        4,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    size: str = typer.Option("small", "--size", "-s", help="Model size"),
    interaction: bool = typer.Option(False, help="Use interaction matrix"),
    local_interaction: bool = typer.Option(False, help="Use local interaction matrix"),
    num_coord: int = typer.Option(
        2, help="Number of features for distance calculation"
    ),
    K: int = typer.Option(10, help="Number of k-neighbors"),
    interaction_type: str = typer.Option("lhc", help="Type of interaction"),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(3, help="Number of global conditioning features"),
    use_pid: bool = typer.Option(False, help="Use particle ID for training"),
    pid_idx: int = typer.Option(4, help="Index of the PID in the input array"),
    pid_dim: int = typer.Option(8, help="Number of unique PIDs"),
    use_add: bool = typer.Option(
        True, help="Use additional features beyond kinematic information"
    ),
    num_add: int = typer.Option(5, help="Number of additional features"),
    zero_add: bool = typer.Option(
        False,
        help="Load the model with additional blocks but zero the inputs from the dataloader",
    ),
    use_clip: bool = typer.Option(False, help="Use CLIP loss during training"),
    use_event_loss: bool = typer.Option(
        False, help="Use additional classification loss between physics process"
    ),
    mode: str = typer.Option(
        "classifier", help="Task to run: classifier, generator, pretrain"
    ),
    regression_loss: str = typer.Option(
        "mse", help="Regression loss type: mse (L2), l1 (MAE), or huber"
    ),
    # Training options
    batch: int = typer.Option(64, help="Batch size"),
    iterations: int = typer.Option(-1, help="Number of iterations per pass"),
    epoch: int = typer.Option(10, help="Number of epochs"),
    warmup_epoch: int = typer.Option(0, help="Number of learning rate warmup epochs"),
    use_amp: bool = typer.Option(False, help="Use amp"),
    clip_inputs: bool = typer.Option(
        False, help="Clip input dataset to be within R=0.8 and atl least 500 MeV"
    ),
    # Optimizer
    optim: str = typer.Option("lion", help="optimizer to use"),
    sched: str = typer.Option("cosine", help="lr scheduler to use"),
    b1: float = typer.Option(0.95, help="Lion b1"),
    b2: float = typer.Option(0.98, help="Lion b2"),
    lr: float = typer.Option(5e-5, help="Learning rate"),
    lr_factor: float = typer.Option(
        1.0, help="Learning rate factor for new layers during fine-tuning"
    ),
    wd: float = typer.Option(0.0, help="Weight decay"),
    nevts: int = typer.Option(-1, help="Maximum number of events to use"),
    event_sampler_random_state: int = typer.Option(42, help="Random state for event sampler"),
    # Model
    attn_drop: float = typer.Option(0.0, help="Dropout for attention layers"),
    mlp_drop: float = typer.Option(0.0, help="Dropout for mlp layers"),
    feature_drop: float = typer.Option(0.0, help="Dropout for input features"),
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
    max_particles: int = typer.Option(150, help="Maximum number of particles per event"),
    class_current_type: bool = typer.Option(False, help="Classify current type"),
    class_pions: bool = typer.Option(False, help="Classify single pion"),
    regress_E_available: bool = typer.Option(False, help="Regress energy available"),
    regress_E_available_no_muon: bool = typer.Option(False, help="Regress energy available without muon"),
    regress_log: bool = typer.Option(False, help="Log transformation for regression"),
    weight_loss: bool = typer.Option(False, help="Weight loss for regression using 1/E weights"),
):
    args = SimpleNamespace(
        mode=mode,
        regress_log=regress_log,
        regress_E_available=regress_E_available,
        regress_E_available_no_muon=regress_E_available_no_muon,
        classification_event_type=False,
        classification_current=class_current_type,
        classification_cc_1pi=False,
        classification_n_pions=False,
        classification_CC1orNPi=class_pions,
    )
    task = create_task(args)
    num_classes = len(task.class_idx_map) if task.class_idx_map is not None else 2
    run_training(
        outdir,
        save_tag,
        pretrain_tag,
        dataset,
        path,
        wandb,
        fine_tune,
        resuming,
        num_feat,
        size,
        interaction,
        local_interaction,
        num_coord,
        K,
        interaction_type,
        conditional,
        num_cond,
        use_pid,
        pid_idx,
        pid_dim,
        use_add,
        num_add,
        zero_add,
        use_clip,
        use_event_loss,
        num_classes,
        2,
        mode,
        batch,
        iterations,
        epoch,
        warmup_epoch,
        use_amp,
        optim,
        sched,
        b1,
        b2,
        lr,
        lr_factor,
        wd,
        nevts,
        event_sampler_random_state,
        attn_drop,
        mlp_drop,
        feature_drop,
        num_workers,
        clip_inputs=clip_inputs,
        regression_loss=regression_loss,
        max_particles=max_particles,
        task=task,
        weight_loss=weight_loss,
    )


@app.command()
def evaluate(
    indir: str = typer.Option(
        "", "--input_dir", "-i", help="Directory to input best model"
    ),
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output evaluation results"
    ),
    batch: int = typer.Option(
        None, "--batch", help="Override evaluation batch size from settings"
    ),
    num_workers: int = typer.Option(
        None,
        "--num-workers",
        help="Override number of dataloader workers from settings",
    ),
):
    if not outdir:
        outdir = os.path.join(indir, "test_results")

    settings = json.load(open(f"{indir}/settings.json", "r"))
    task = Task(**settings["task"])
    num_classes = (
        len(task.class_idx_map)
        if task.class_idx_map is not None
        else settings.get("num_classes", 2)
    )
    run_evaluation(
        indir,
        outdir,
        settings.get("save_tag", ""),
        settings["dataset"],
        settings["path"],
        settings["num_feat"],
        settings["model_size"],
        settings["interaction"],
        settings["local_interaction"],
        settings["num_coord"],
        settings["K"],
        settings["interaction_type"],
        settings["conditional"],
        settings["num_cond"],
        settings["use_pid"],
        settings["pid_idx"],
        settings.get("pid_dim", 9),
        settings["use_add"],
        settings["num_add"],
        settings["use_event_loss"],
        num_classes,
        settings.get("num_gen_classes", 1),
        settings["mode"],
        batch if batch is not None else settings["batch"],
        num_workers if num_workers is not None else settings["num_workers"],
        clip_inputs=settings.get("clip_inputs", False),
        max_particles=settings.get("max_particles", 150),
        task=task,
    )



@app.command()
def dataloader(
    dataset: str = typer.Option(
        "top", "--dataset", "-d", help="Dataset name to download"
    ),
    folder: str = typer.Option(
        "./", "--folder", "-f", help="Folder to save the dataset"
    ),
):
    for tag in ["train", "test", "val"]:
        print(tag)
        load_data(dataset, folder, dataset_type=tag, distributed=False)


if __name__ == "__main__":
    app()
