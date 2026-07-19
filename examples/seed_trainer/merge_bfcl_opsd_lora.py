"""Merge a SEED LoRA checkpoint and verify that vLLM can reload it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def latest_adapter(checkpoint_root: Path) -> tuple[Path, Path]:
    steps = sorted(
        checkpoint_root.glob("global_step_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not steps:
        raise FileNotFoundError(f"no resumable SEED checkpoint under {checkpoint_root}")
    step = steps[-1]
    adapter = step / "actor" / "lora_adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter is missing from {step}")
    return step, adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--validate-vllm", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    step, adapter = latest_adapter(checkpoint_root)
    checksum_evidence = checkpoint_root.parent.parent / "evidence" / checkpoint_root.name / "trainable_checksum.json"
    if not checksum_evidence.is_file():
        raise FileNotFoundError(f"missing trainable checksum evidence: {checksum_evidence}")
    checksum_value = json.loads(checksum_evidence.read_text(encoding="utf-8"))
    if checksum_value.get("changed") is not True:
        raise RuntimeError("LoRA trainable checksum did not change")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    peft_model = PeftModel.from_pretrained(base, adapter)
    merged = peft_model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False).save_pretrained(output)
    del peft_model, merged, base

    reload_result = None
    if args.validate_vllm:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(output),
            tensor_parallel_size=1,
            gpu_memory_utilization=0.45,
            max_model_len=args.max_model_len,
            enforce_eager=True,
            trust_remote_code=False,
        )
        generated = llm.generate(
            ["<|im_start|>user\nReply OK.<|im_end|>\n<|im_start|>assistant\n"],
            SamplingParams(temperature=0.0, max_tokens=2),
        )
        reload_result = {
            "ok": bool(generated and generated[0].outputs),
            "text": generated[0].outputs[0].text if generated and generated[0].outputs else "",
        }
        if not reload_result["ok"]:
            raise RuntimeError("merged Hugging Face export failed vLLM reload")

    report = {
        "checkpoint": str(step),
        "checkpoint_resumable": (step / "actor").is_dir() and (step / "data.pt").is_file(),
        "adapter": str(adapter),
        "adapter_sha256": tree_sha256(adapter),
        "trainable_checksum": checksum_value,
        "merged_export": str(output),
        "merged_export_sha256": tree_sha256(output),
        "vllm_reload": reload_result,
    }
    if not report["checkpoint_resumable"]:
        raise RuntimeError("checkpoint is not resumable")
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SEED_BFCL_EXPORT_OK checkpoint={step} export={output}")


if __name__ == "__main__":
    main()
