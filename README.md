```
pip install "jax[cpu]===0.4.10" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install --upgrade "jax[cpu]"
pip install git+https://github.com/deepmind/dm-haiku

pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  --proxy='socks5://127.0.0.1:1080'
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --proxy='socks5://127.0.0.1:1080'
```