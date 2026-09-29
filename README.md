# Verus: Deepfake Detection

This project serves the existing website and analyzes a visible face in a JPG, PNG, WebP, MP4, WebM, or MOV upload. The API returns a GenD fake score for the largest detected face in an image or in up to eight sampled video frames. Files are written to a temporary location during analysis and deleted after the response.

`frontend/` contains the website and images. `backend/` contains the API and model files. `tests/` contains API checks.

## Run locally

Python 3.12 is recommended. The pretrained GenD checkpoint is 1.22 GB and is deliberately excluded from version control.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
curl -fL -o backend/models/gend-clip-l14.safetensors 'https://huggingface.co/yermandy/GenD_CLIP_L_14/resolve/891ce014a0308386c4d7d25b3dcf436a22db5504/model.safetensors'
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

For later runs, double-click `start.command` on macOS or run `./start.command` in a terminal. Open <http://127.0.0.1:8000> in your browser. Keep the server terminal open during analysis. Opening `frontend/index.html` directly or using a separate static server will not connect the page to the API.

Choose a file and press **Run detection**. The classification appears directly below the media preview as **LIKELY REAL**, **LIKELY FAKE**, or **INCONCLUSIVE**, alongside the fake signal score. These are model estimates, not proof of authenticity. The API is at `POST /api/analyze`, with multipart field `file`; API docs are at <http://127.0.0.1:8000/api/docs>. `GET /api/health` reports whether the detector can load and is ready.

The downloaded checkpoint's SHA-256 is `d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833`. The bundled YuNet model's SHA-256 is `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.

## Validate and calibrate

The repository includes thresholds validated on the HiDF workflow below. Production mode (`VERITY_PRODUCTION=1`) refuses analysis if `backend/models/thresholds.json` is missing or does not match the model, face detector, preprocessing files, and inference code used during validation. The health endpoint loads the complete detector before reporting ready.

### HiDF calibration and held-out validation

The public [HiDF video dataset](https://zenodo.org/records/16140829) is available for noncommercial research under [CC BY-NC 4.0](https://github.com/DSAIL-SKKU/HiDF#request-for-hidf). Its real and manipulated videos are paired by source ID. Download a reproducible 500-pair sample, assign 250 pairs to calibration and 250 pairs to held-out testing, then validate:

```sh
.venv/bin/python -m scripts.hidf_sample --pairs 500 --calibration-pairs 250 --out datasets/HiDF
.venv/bin/python -m scripts.validate score datasets/HiDF/manifest.csv evaluation/hidf_scores.csv --root datasets/HiDF
.venv/bin/python -m scripts.validate calibrate evaluation/hidf_scores.csv evaluation/hidf_report.json backend/models/thresholds.json
```

The sampler verifies the ZIP CRC for every downloaded video, records its seed and selection in `datasets/HiDF/selection.json`, and keeps connected base and target identities in one split. Both splits contain 250 real and 250 fake videos and exceed the validation requirement of 100 distinct identity groups per class. The media, per-file scores, and detailed report remain outside version control.

The reproducible validation completed on 2026-09-29 using seed `20260928` and the Linux production image. It selected thresholds of **0.4182** and **0.8962**. On the 500 held-out videos, **313** results were correct, **178** were inconclusive, and **9** were wrong. Accuracy among the 322 conclusive results was **97.2%**. All videos contained a detectable face; the one-sided 95% source-group error bounds were **4.13%** for missed fakes and **3.59%** for false alarms. These thresholds validate this pipeline for HiDF-style face-swap videos in a noncommercial research deployment. They do not establish performance on images, audio, fully generated faces, unseen manipulation methods, or arbitrary public uploads.

### FaceForensics++

The project is prepared for [FaceForensics++](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md). Its videos require [maintainer approval through this form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform). The [dataset terms](https://kaldir.vc.in.tum.de/faceforensics_tos.pdf) permit non-commercial research and education. Review and accept them yourself; the download script is sent after approval. For a commercial deployment, obtain a dataset with suitable rights. No FaceForensics++ videos are bundled here.

The released GenD model was trained on FaceForensics++ `c23` **training** videos. This workflow uses only the official `val.json` and `test.json` source IDs, checks that neither overlaps `train.json`, calibrates on validation videos, and reports results on test videos. `scripts/faceforensics_manifest.py` fetches and caches these public split files; it never fetches the protected videos.

Once you have the approved downloader, obtain `c23` videos under `datasets/FaceForensics++`. Replace the placeholder path with the script sent by the maintainers, and handle its terms prompt yourself:

```sh
for part in original Deepfakes Face2Face FaceSwap NeuralTextures; do
  python /path/to/approved/download-FaceForensics.py datasets/FaceForensics++ -d "$part" -c c23 -t videos --server EU
done
```

Then run:

```sh
.venv/bin/python -m scripts.faceforensics_manifest datasets/FaceForensics++ datasets/faceforensics_manifest.csv
.venv/bin/python -m scripts.validate score datasets/faceforensics_manifest.csv evaluation/faceforensics_scores.csv --root datasets/FaceForensics++
.venv/bin/python -m scripts.validate calibrate evaluation/faceforensics_scores.csv evaluation/faceforensics_report.json backend/models/thresholds.json
```

The manifest generator requires the expected videos for every source ID and method. It fails with the missing paths if the download is incomplete. It groups each real video with fakes targeting the same source, so related files remain in one split. To evaluate a single method first, pass `--methods Deepfakes` to the manifest command. The full four-method run is needed to check each manipulation type.

Use at least **100 detectable real and 100 detectable fake files in each split**, from at least **100 distinct source groups per class per split**. The validation script requires at least 20% correct-class coverage across all labeled media, including files with no detectable face, at most 10% files with no detectable face, and a 95% one-sided source-group error upper bound of at most 5% for both false-real and false-fake classifications. It also checks held-out fake results by manipulation method, reports inconclusive results, and breaks out optional `slice` values. These checks measure performance on FaceForensics++ `c23` videos. They do not establish performance on new methods, different compression, fully generated faces, or arbitrary public uploads; use independent current media before making broader claims.

For another labeled dataset, create a CSV with `path,label,split,group,slice,method` columns and run the same `score` and `calibrate` commands with its root path.

```csv
path,label,split,group,slice,method
real/source_001.mp4,real,calibration,source_001,mp4-compressed,original
fake/source_001_swap.mp4,fake,calibration,source_001,mp4-compressed,face-swap
```

The scoring command uses CPU inference. Run it on the Linux deployment host or inside the supplied image because operating-system video decoders can select slightly different pixels. Run production inference on CPU with these thresholds. Inspect `evaluation/faceforensics_report.json` before deployment. A failed scoring or calibration run removes stale output at the requested path. A failed calibration exits nonzero and removes any previous thresholds there. Both the datasets and generated reports are excluded from version control. Restart the server after changing thresholds.

## Deploy on a Docker VPS

The included `Dockerfile`, `compose.yaml`, and `Caddyfile` set up one CPU inference worker behind Caddy with HTTPS, an upload body limit, health checks, and a read-only app filesystem. Use an AMD64 Linux VPS with at least 8 GB RAM and 20 GB free disk space, plus a domain pointing to it. Open ports 80 and 443. Keep port 8000 private; Compose does not publish it.

1. Install Docker Engine and the Compose plugin on the VPS, and copy this project to it. Obtain the GenD checkpoint using the command above and verify its SHA-256 with `sha256sum backend/models/gend-clip-l14.safetensors`.
2. The included `backend/models/thresholds.json` supports the validated noncommercial HiDF research deployment. For another use case, run the validation workflow with representative labeled media and replace that file. It is bound to the exact model, face detector, preprocessing files, and inference code used during validation.
3. Set `VERITY_DOMAIN` to your DNS hostname and start the stack:

   ```sh
   export VERITY_DOMAIN=detector.example.com
   docker compose up -d --build
   curl -fsS https://detector.example.com/api/health
   ```

The health response must include `"ready":true` and `"calibrated":true`. Check `docker compose logs app proxy` if it does not. Caddy obtains and renews TLS certificates for the configured hostname. The app accepts anonymous uploads; for a public launch, configure request-rate controls at your host or edge provider and monitor CPU, memory, error rates, and false-result reports. Review the privacy policy and dataset rights for your actual deployment jurisdiction. Do not use this detector as the sole basis for consequential decisions.

## Deploy on Vercel

`Dockerfile.vercel` packages the same CPU service as a Vercel container Function, including the verified checkpoint and calibrated thresholds. Create a new Vercel project from this repository with Fluid compute enabled. New projects support large Functions automatically; an existing project must set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` before redeploying. Vercel limits Function request bodies to 4.5 MB, so the browser rejects files over 4 MB on this deployment.

## Model choice and limits

[GenD CLIP ViT-L/14](https://github.com/yermandy/GenD) is a pretrained face manipulation detector from [WACV 2026 research](https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf), evaluated across 14 benchmarks. Its [released weights](https://huggingface.co/yermandy/GenD_CLIP_L_14) and code are MIT licensed. [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) supplies face landmarks for alignment; its bundled model is MIT licensed (see `backend/models/LICENSE-YUNET`).

There is no universal best detector for every new deepfake technique. GenD was trained on face crops for face manipulations and can miss entirely AI-generated faces. This implementation uses YuNet rather than the paper's RetinaFace preprocessing, so its scores may differ from published results. The included thresholds are calibrated only for the documented HiDF video workflow. A score is not a probability that the media is authentic. A low fake score does not prove a file is real.

Video analysis sequentially decodes at most the first 240 frames and averages scores from up to eight evenly spaced frames. It does **not** inspect audio or run a temporal neural network. Only the largest face in each sampled frame is checked. Media with no detectable face returns `inconclusive`. Validate on your own real, manipulated, compressed, and demographic test sets before relying on a result in a consequential setting.

## Check

```sh
.venv/bin/python -m unittest discover -s tests
node --check frontend/script.js
```
