"""Local msgpack HTTP policy service; one process per PPU."""
import argparse
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import traceback

import torch
from deployment.model_server.tools import msgpack_numpy
from DiT4DiT.model.framework.base_framework import baseframework
from .common import file_sha
from .data import CachedLibero
from .donors import DonorIndex
from .interventions import ATTRIBUTION_CONDITIONS


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--smoke", action="store_true", help="Allow a partial cache for explicit runtime tests")
    p.add_argument("--protocol", help="Frozen evaluation protocol and provenance JSON")
    args = p.parse_args()
    model = baseframework.from_pretrained(args.checkpoint).cuda().eval()
    if model.config.framework.name == "DiT4DiTQwenJoint":
        from .current_images import CurrentImageLibero
        donor_dataset = CurrentImageLibero(model.config, allow_partial=args.smoke)
    else:
        donor_dataset = CachedLibero(model.config, allow_partial=args.smoke)
    donor_index = DonorIndex(donor_dataset)
    if len(model.norm_stats) != 1:
        raise ValueError("Expected the upstream single FRANKA normalization group")
    metadata = {"checkpoint_sha256": file_sha(args.checkpoint),
                "action_stats": next(iter(model.norm_stats.values()))["action"]}
    protocol = json.loads(Path(args.protocol).read_text()) if args.protocol else {}
    if protocol:
        if protocol["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("Evaluation protocol refers to a different checkpoint")
        metadata["protocol_sha256"] = file_sha(args.protocol)
        metadata["observation_mode"] = protocol["observation_mode"]
        metadata["noise_std"] = protocol["noise_std"]
    initial_cache = OrderedDict()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(metadata).encode())

        def do_POST(self):
            try:
                request = msgpack_numpy.unpackb(self.rfile.read(int(self.headers["Content-Length"])))
                condition = request["intervention"]
                if condition not in ATTRIBUTION_CONDITIONS:
                    raise ValueError(f"Unknown condition: {condition}")
                if protocol:
                    if request.get("protocol_sha256") != metadata["protocol_sha256"]:
                        raise ValueError("Client/server evaluation protocols differ")
                    if condition not in protocol["conditions"]:
                        raise ValueError("Condition is outside the frozen evaluation protocol")
                    example = request["example"]
                    if len(example["image"]) != 1 or any(k in example for k in ("target_mean", "cond_mean", "qwen")):
                        raise ValueError("Only the observed current image is permitted at inference")
                donor = None
                if condition.startswith("swap"):
                    donor = [donor_index.select(request["example"]["lang"], request["episode_seed"], request["progress"])]
                initial_encoded = None
                if condition.startswith("freeze_"):
                    initial = request["initial_example"]
                    if len(initial["image"]) != 1 or initial["lang"] != request["example"]["lang"]:
                        raise ValueError("Initial observation must have one frame and the same instruction")
                    key = (request["episode_seed"], initial["lang"],
                           hashlib.sha256(initial["image"][0].tobytes()).hexdigest())
                    if key not in initial_cache:
                        initial_cache[key] = model.observations([initial])
                        if len(initial_cache) > 8:
                            initial_cache.popitem(last=False)
                    initial_cache.move_to_end(key)
                    initial_encoded = initial_cache[key]
                result = model.predict_action([request["example"]], intervention=condition, donor_examples=donor,
                    feature_seed=request.get("feature_seed", 42), action_seed=request["action_seed"],
                    noise_std=protocol.get("noise_std", 0.1), initial_encoded=initial_encoded)
                body = msgpack_numpy.packb(result)
                self.send_response(200)
            except Exception:
                body = traceback.format_exc().encode()
                traceback.print_exc()
                self.send_response(500)
            self.end_headers()
            self.wfile.write(body)

    print(f"READY http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
