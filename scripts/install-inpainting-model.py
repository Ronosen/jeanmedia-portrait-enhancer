"""Install the optional 208 MB model only with explicit download consent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

URL="https://huggingface.co/Carve/LaMa-ONNX/resolve/a3ee2fca54baebec351b8fa7786154ffa7555aa6/lama_fp32.onnx"
EXPECTED="1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"


def main():
    parser=argparse.ArgumentParser(description="208 MB local LaMa inpainting model; photos are never uploaded.")
    parser.add_argument("--consent",action="store_true",help="Explicitly approve this 208 MB download")
    args=parser.parse_args()
    if not args.consent:
        parser.error("Consent is required before downloading the 208 MB model.")
    directory=Path(__file__).resolve().parents[1]/"worker/models"
    target=directory/"lama_fp32.onnx"
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=EXPECTED:
            raise ValueError("An existing model differs; it was not overwritten.")
        print("Verified existing model.");return
    descriptor,temporary=tempfile.mkstemp(prefix="lama-",suffix=".download",dir=directory)
    digest=hashlib.sha256();total=0
    try:
        request=urllib.request.Request(URL,headers={"User-Agent":"JeanMedia-model-installer/1.0"})
        with os.fdopen(descriptor,"wb") as stream,urllib.request.urlopen(request,timeout=60) as response:
            while block:=response.read(1024*1024):
                total+=len(block)
                if total>230_000_000:raise ValueError("The download exceeded its expected size.")
                stream.write(block);digest.update(block)
                if total%(10*1024*1024)==0:print(f"Downloaded {total//(1024*1024)} MiB",flush=True)
        if digest.hexdigest()!=EXPECTED:raise ValueError("The model failed SHA256 verification.")
        os.link(temporary,target)
        (directory/"LaMa-PROVENANCE.json").write_text(json.dumps({
            "source":URL,"sha256":EXPECTED,"bytes":total,"declaredLicense":"Apache-2.0",
            "modelCard":"https://huggingface.co/Carve/LaMa-ONNX",
            "originalSource":"https://github.com/advimman/lama",
            "releaseLicenseAudit":"pending; publisher declaration is not commercial certification"},indent=2),encoding="utf-8")
        print("Installed and SHA256 verified.")
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__=="__main__":main()
