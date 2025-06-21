import torch, os, json
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, VideoDataset_pt, ModelLogger, launch_training_task, wan_parser, launch_data_process_task
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        resume_from_checkpoint=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]

        # print(f"Loading models: {model_configs}");assert 0
        if resume_from_checkpoint is None: 
            self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        else:
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
                model_configs=[
                    ModelConfig(path=['models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors', resume_from_checkpoint], offload_device=None),
                    ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device=None),
                    ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device=None),
                ],
            )
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
        self.model_input_keys = ['input_video', 'height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'vace_video', 'vace_reference_image', 'noise', 'latents', 'input_latents', 'vace_context', 'prompt', 'context']
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        if 0:
            print(f"Input video size: {inputs_shared['input_video'][0].size}")
            print(f"Input video num frames: {inputs_shared['num_frames']}")
            print(f"Input video height: {inputs_shared['height']}")
            print(f"Input video width: {inputs_shared['width']}")
            for extra_input in self.extra_inputs:
                # check if extra_input is image or list of images
                if extra_input in inputs_shared:
                    if isinstance(inputs_shared[extra_input], list):
                        print(f"Extra input {extra_input}: {[img.size for img in inputs_shared[extra_input]]}")
                    elif isinstance(inputs_shared[extra_input], torch.Tensor):
                        print(f"Extra input {extra_input}: {inputs_shared[extra_input].size()}")
                    else:
                        print(f"Extra input {extra_input}: {inputs_shared[extra_input]}")
            assert 0

            # Input video size: (832, 480) 
            # Input video num frames: 49       
            # Input video height: 480 
            # Input video width: 832 
            # Extra input vace_video: [(832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480), (832, 480)]
            # Extra input vace_reference_image: <PIL.Image.Image image mode=RGB size=832x480 at 0x7FB27AF4A460>

            # print(inputs_shared.keys(), inputs_posi.keys(), inputs_nega.keys());assert 0 # 
            # dict_keys(['input_video', 'height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'vace_video', 'vace_reference_image']) dict_keys(['prompt']) dict_keys([])


        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        # print(inputs_shared.keys(), inputs_posi.keys(), inputs_nega.keys());assert 0 # 
        # dict_keys(['input_video', 'height', 'width', 'num_frames', 'cfg_scale', 'tiled', 'rand_device', 'use_gradient_checkpointing', 'use_gradient_checkpointing_offload', 'cfg_merge', 'vace_scale', 'vace_video', 'vace_reference_image', 'noise', 'latents', 'input_latents', 'vace_context']) dict_keys(['prompt', 'context']) dict_keys(['context'])

        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        # print(inputs.keys());assert 0  # Debugging line to check inputs and models
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = VideoDataset(args=args) if args.use_data_pt is None else VideoDataset_pt(args=args)
    
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    assert not (args.use_data_pt is not None and args.data_process), \
        "You must choose one of --use_data_pt or --data_process, not both."
        
    if args.data_process:
        # Launch data processing task
        launch_data_process_task(
            model, dataset, args.output_path
        )
    else:
        launch_training_task(
            dataset, model, model_logger, optimizer, scheduler,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_data_pt=args.use_data_pt,
            args=args,
        )
