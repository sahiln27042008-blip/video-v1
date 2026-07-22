`POST /jobs` – upload video, get job_id

`POST /jobs`
`GET /jobs/{id}/status` – check progress

`GET /jobs/{id}/status`
`GET /jobs/{id}/timeline` – download timeline.json

`GET /jobs/{id}/timeline`
`GET /jobs/{id}/timeline_words` – download timeline_with_words.json

`GET /jobs/{id}/timeline_words`
`POST /shutdown` – stop the worker

`POST /shutdown`
Jobs are checkpointed after each module. If the worker crashes, rerunning the job resumes from the last successful step.

Models are cached in memory and reused across jobs.

GPU acceleration for WhisperX, InsightFace, and PyTorch.

write_file("colab_worker/__init__.py", "")
write_file("local_client/__init__.py", "")

print("✅ All files created in ./remote_gpu/")
print("To use, first run:\n   python create_remote_gpu.py   (this already done)")
print("\nNext steps:")
print("1. Upload the 'remote_gpu' folder to Colab or run the notebook.")
print("2. On local machine, set REMOTE_GPU_URL and use local_client.")