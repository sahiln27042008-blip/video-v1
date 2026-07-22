`POST /jobs` – upload video

`POST /jobs`
`GET /jobs/{id}/status` – get progress

`GET /jobs/{id}/status`
`GET /jobs/{id}/timeline` – download timeline.json

`GET /jobs/{id}/timeline`
`GET /jobs/{id}/timeline_words` – download timeline_with_words.json

`GET /jobs/{id}/timeline_words`
`POST /shutdown` – stop worker

`POST /shutdown`
Jobs checkpoint after each module. Resume after crash.
""")

write_file("colab_worker/__init__.py", "")
write_file("local_client/__init__.py", "")

print("\n" + "="_60)
print("All remote GPU files generated successfully!")
print("Location: ./remote_gpu/")
print("="_60)
print("\nNext steps:")
print("1. Ensure .env exists in project root with HF_TOKEN")
print("2. cd remote_gpu && python -m colab_worker.main")
print("3. Submit jobs via remote_gpu.local_client.client.RemoteGPUClient")