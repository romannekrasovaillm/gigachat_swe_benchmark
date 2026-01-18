#!/usr/bin/env python3
"""
Run SWE-bench evaluation WITHOUT containers (LocalEnvironment).
ASYNC VERSION - processes multiple instances in parallel.

WARNING: This runs code directly on your machine without isolation!
Use only for testing purposes.

This script:
1. Downloads SWE-bench instances
2. Clones repositories locally (in parallel)
3. Applies base commits
4. Runs the agent in LocalEnvironment (in parallel with semaphore)
5. Collects patches
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

# Try to import from pip-installed package first, fall back to local path
try:
    from minisweagent.agents.default import DefaultAgent
except ImportError:
    # Fall back to local path if running from mini-swe-agent repo
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def setup_logging(output_dir: Path):
    """Setup logging to file and console."""
    log_file = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('swebench')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def truncate_text(text: str, max_len: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [{len(text) - max_len} chars truncated]"


class LoggingAgent(DefaultAgent):
    """Agent with logging of model responses."""

    def __init__(self, *args, logger=None, instance_id="", quiet=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger or logging.getLogger('swebench')
        self.instance_id = instance_id
        self.step_count = 0
        self.quiet = quiet

    def step(self) -> dict:
        """Override step to log model responses."""
        self.step_count += 1
        self.logger.info(f"[{self.instance_id}] Step {self.step_count} starting...")
        return super().step()

    def query(self) -> dict:
        """Override query to log model response."""
        response = super().query()
        content = response.get("content", "")

        if not self.quiet:
            print(f"\n{'='*60}")
            print(f"[{self.instance_id}] STEP {self.step_count} - MODEL RESPONSE:")
            print("-" * 60)
            print(content[:2000] if len(content) > 2000 else content)
            if len(content) > 2000:
                print(f"... [{len(content) - 2000} chars truncated]")
            print("=" * 60)

        self.logger.info(f"[{self.instance_id}] Step {self.step_count} MODEL RESPONSE ({len(content)} chars)")
        self.logger.debug(f"[{self.instance_id}] Full response:\n{content}")

        return response

    def parse_action(self, response: dict) -> dict:
        """Override parse_action to log extracted command."""
        import re
        actions = re.findall(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)

        if len(actions) == 1:
            cmd = actions[0].strip()
            if not self.quiet:
                print(f"\n>>> [{self.instance_id}] EXTRACTED COMMAND:")
                print("-" * 40)
                print(cmd[:500] if len(cmd) > 500 else cmd)
                if len(cmd) > 500:
                    print(f"... [{len(cmd) - 500} chars truncated]")
                print("-" * 40)
            self.logger.info(f"[{self.instance_id}] Extracted command: {cmd[:200]}...")
            return {"action": cmd, **response}

        if not self.quiet:
            print(f"\n[!] [{self.instance_id}] FORMAT ERROR: Found {len(actions)} bash blocks (expected 1)")
            if actions:
                for i, a in enumerate(actions):
                    print(f"  Block {i+1}: {a[:100]}...")
        self.logger.warning(f"[{self.instance_id}] Format error: {len(actions)} bash blocks")

        from minisweagent.agents.default import FormatError
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        """Override execute_action to log output."""
        import subprocess
        try:
            output = self.env.execute(action["action"])

            out_text = output.get("output", "")
            if not self.quiet:
                print(f"\n<<< [{self.instance_id}] COMMAND OUTPUT ({len(out_text)} chars):")
                print("-" * 40)
                print(out_text[:1000] if len(out_text) > 1000 else out_text)
                if len(out_text) > 1000:
                    print(f"... [{len(out_text) - 1000} chars truncated]")
                print("-" * 40)
            self.logger.info(f"[{self.instance_id}] Command output: {len(out_text)} chars")
            self.logger.debug(f"[{self.instance_id}] Full output:\n{out_text}")

        except subprocess.TimeoutExpired as e:
            out = e.output.decode("utf-8", errors="replace") if e.output else ""
            if not self.quiet:
                print(f"\n[!] [{self.instance_id}] COMMAND TIMEOUT")
            self.logger.warning(f"[{self.instance_id}] Command timeout")
            from minisweagent.agents.default import ExecutionTimeoutError
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=out)
            )
        except TimeoutError:
            if not self.quiet:
                print(f"\n[!] [{self.instance_id}] COMMAND TIMEOUT")
            self.logger.warning(f"[{self.instance_id}] Command timeout")
            from minisweagent.agents.default import ExecutionTimeoutError
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output="")
            )

        self.has_finished(output)
        return output


async def run_subprocess(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> tuple[int, str, str]:
    """Run subprocess asynchronously."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}")


async def setup_repo_async(instance: dict, work_dir: Path, logger) -> Path:
    """Clone repo and checkout base commit asynchronously."""
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    instance_id = instance["instance_id"]

    # Use unique directory per instance to avoid conflicts
    repo_dir = work_dir / f"{repo.replace('/', '_')}_{instance_id.replace('/', '_')}"

    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    logger.info(f"[{instance_id}] Cloning {repo}...")
    returncode, stdout, stderr = await run_subprocess(
        ["git", "clone", f"https://github.com/{repo}.git", str(repo_dir)],
        timeout=600
    )
    if returncode != 0:
        raise RuntimeError(f"Git clone failed: {stderr}")

    logger.info(f"[{instance_id}] Checking out {base_commit[:8]}...")
    returncode, stdout, stderr = await run_subprocess(
        ["git", "checkout", base_commit],
        cwd=repo_dir,
        timeout=60
    )
    if returncode != 0:
        raise RuntimeError(f"Git checkout failed: {stderr}")

    # Try to install dependencies (best effort)
    logger.info(f"[{instance_id}] Installing dependencies...")
    try:
        if (repo_dir / "requirements.txt").exists():
            await run_subprocess(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                cwd=repo_dir,
                timeout=120
            )
        if (repo_dir / "setup.py").exists():
            await run_subprocess(
                [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
                cwd=repo_dir,
                timeout=120
            )
        elif (repo_dir / "pyproject.toml").exists():
            await run_subprocess(
                [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
                cwd=repo_dir,
                timeout=120
            )
    except Exception as e:
        logger.warning(f"[{instance_id}] Failed to install deps: {e}")

    return repo_dir


async def get_patch_async(repo_dir: Path) -> str:
    """Get git diff of changes made asynchronously."""
    returncode, stdout, stderr = await run_subprocess(
        ["git", "diff"],
        cwd=repo_dir,
        timeout=60
    )
    return stdout


def run_agent_sync(
    instance: dict,
    config: dict,
    model_name: str | None,
    repo_dir: Path,
    logger,
    quiet: bool = False,
) -> tuple[str, int, str]:
    """Run agent synchronously (to be called in thread pool)."""
    instance_id = instance["instance_id"]

    # Create model and environment
    model = get_model(model_name, config=config.get("model", {}))

    env_config = config.get("environment", {}).copy()
    env_config["cwd"] = str(repo_dir)
    env_config.pop("executable", None)
    env_config.pop("image", None)

    env = LocalEnvironment(**env_config)

    task = instance["problem_statement"]

    logger.info(f"[{instance_id}] TASK preview: {truncate_text(task, 500)}")
    logger.debug(f"[{instance_id}] Full task:\n{task}")

    agent_config = config.get("agent", {})
    agent = LoggingAgent(
        model, env,
        logger=logger,
        instance_id=instance_id,
        quiet=quiet,
        **agent_config
    )

    logger.info(f"[{instance_id}] Agent starting...")
    exit_status, agent_result = agent.run(task)

    return exit_status, model.n_calls, ""


async def process_instance_async(
    instance: dict,
    config: dict,
    model_name: str | None,
    work_dir: Path,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    logger,
    quiet: bool = False,
) -> dict:
    """Process a single SWE-bench instance asynchronously."""
    instance_id = instance["instance_id"]

    result = {
        "instance_id": instance_id,
        "model_name_or_path": model_name or config.get("model", {}).get("model_name", "unknown"),
        "model_patch": "",
        "exit_status": "error",
        "steps": 0,
        "error": None,
    }

    repo_dir = None
    try:
        # Setup repository (can run in parallel)
        repo_dir = await setup_repo_async(instance, work_dir, logger)

        # Acquire semaphore for model inference (limited concurrency)
        async with semaphore:
            logger.info(f"[{instance_id}] Starting agent (semaphore acquired)...")

            if not quiet:
                print(f"\n{'='*60}")
                print(f"Processing: {instance_id}")
                print(f"{'='*60}")

            # Run agent in thread pool (it's synchronous)
            exit_status, n_calls, _ = await asyncio.to_thread(
                run_agent_sync,
                instance,
                config,
                model_name,
                repo_dir,
                logger,
                quiet,
            )

            # Get patch
            patch = await get_patch_async(repo_dir)

            result["model_patch"] = patch
            result["exit_status"] = exit_status
            result["steps"] = n_calls

            logger.info(f"[{instance_id}] Completed: status={exit_status}, steps={n_calls}, patch_size={len(patch)}")

            if not quiet:
                print(f"\n[{instance_id}] Completed: status={exit_status}, steps={n_calls}, patch={len(patch)} chars")

    except Exception as e:
        logger.error(f"[{instance_id}] Error: {e}")
        logger.debug(traceback.format_exc())
        result["error"] = str(e)
        result["exit_status"] = type(e).__name__
        if not quiet:
            print(f"\n[{instance_id}] Error: {e}")

    finally:
        # Save trajectory
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        traj_file = instance_dir / f"{instance_id}.traj.json"
        traj_file.write_text(json.dumps({
            "instance_id": instance_id,
            "exit_status": result["exit_status"],
            "steps": result["steps"],
            "error": result.get("error"),
        }, indent=2))

        # Cleanup repo
        if repo_dir and repo_dir.exists():
            try:
                shutil.rmtree(repo_dir)
            except Exception:
                pass

    return result


async def main_async():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench locally WITHOUT containers - ASYNC VERSION"
    )
    parser.add_argument(
        "--subset", default="lite",
        help="SWE-bench subset (lite, verified, full)"
    )
    parser.add_argument(
        "--split", default="dev",
        help="Dataset split"
    )
    parser.add_argument(
        "--slice", default="",
        help="Instance slice (e.g., 0:5)"
    )
    parser.add_argument(
        "--filter", default="",
        help="Filter instance IDs by regex"
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help="Model name override"
    )
    parser.add_argument(
        "-c", "--config",
        default=Path(__file__).parent / "gigachat_swebench_local.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "-o", "--output",
        default=Path(__file__).parent / "gigachat_local_results",
        help="Output directory"
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for repos (default: temp dir)"
    )
    parser.add_argument(
        "-j", "--concurrency", type=int, default=1,
        help="Number of parallel instances to process (default: 1)"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Quiet mode - reduce output verbosity"
    )

    args = parser.parse_args()

    print("="*60)
    print("  SWE-bench Local Runner - ASYNC VERSION")
    print("  WARNING: Code runs directly on your machine!")
    print(f"  Concurrency: {args.concurrency}")
    print("="*60)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = yaml.safe_load(config_path.read_text())

    # Load dataset
    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    print(f"\nLoading dataset: {dataset_path}, split: {args.split}")
    instances = list(load_dataset(dataset_path, split=args.split))

    # Filter instances
    if args.filter:
        import re
        instances = [i for i in instances if re.match(args.filter, i["instance_id"])]

    if args.slice:
        parts = [int(x) if x else None for x in args.slice.split(":")]
        instances = instances[slice(*parts)]

    print(f"Running on {len(instances)} instances with concurrency={args.concurrency}")

    # Setup directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger, log_file = setup_logging(output_dir)
    logger.info(f"Starting SWE-bench run: {len(instances)} instances, concurrency={args.concurrency}")
    logger.info(f"Subset: {args.subset}, Split: {args.split}")
    print(f"Log file: {log_file}")

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="swebench_local_"))

    print(f"Output: {output_dir}")
    print(f"Work dir: {work_dir}")

    # Create semaphore for limiting concurrent model calls
    semaphore = asyncio.Semaphore(args.concurrency)

    # Create tasks for all instances
    tasks = [
        process_instance_async(
            instance,
            config,
            args.model,
            work_dir,
            output_dir,
            semaphore,
            logger,
            quiet=args.quiet or args.concurrency > 1,  # Auto-quiet in parallel mode
        )
        for instance in instances
    ]

    # Process all instances
    preds = {}
    results = []

    # Use as_completed to save results incrementally
    print(f"\nProcessing {len(tasks)} instances...")
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1

        preds[result["instance_id"]] = {
            "model_name_or_path": result["model_name_or_path"],
            "instance_id": result["instance_id"],
            "model_patch": result["model_patch"],
        }

        # Save predictions incrementally
        (output_dir / "preds.json").write_text(json.dumps(preds, indent=2))

        # Progress
        status = result["exit_status"]
        has_patch = "✓" if result["model_patch"] else "✗"
        print(f"[{completed}/{len(tasks)}] {result['instance_id']}: {status} patch:{has_patch}")
        logger.info(f"Progress: {completed}/{len(tasks)} completed")

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    completed_ok = sum(1 for r in results if r["exit_status"] == "done")
    with_patch = sum(1 for r in results if r["model_patch"])

    print(f"Total: {len(results)}")
    print(f"Completed: {completed_ok} ({completed_ok/len(results)*100:.1f}%)")
    print(f"With patch: {with_patch} ({with_patch/len(results)*100:.1f}%)")
    print(f"\nResults saved to: {output_dir}")
    print(f"Log file: {log_file}")

    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Completed: {completed_ok} ({completed_ok/len(results)*100:.1f}%)")
    logger.info(f"With patch: {with_patch} ({with_patch/len(results)*100:.1f}%)")
    logger.info("="*60)

    # Cleanup work dir if temp
    if not args.work_dir:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
