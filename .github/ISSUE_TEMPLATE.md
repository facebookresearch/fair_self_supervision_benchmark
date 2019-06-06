## Guideline for posting issues

If you have a feature request or a bug report, please open the github issue an
fill the template below. If you have general questions the benchmark, please
reach out to prigoyal@fb.com.

Please add a prefix [feature] or [bug] at the beginning of issue title to help
with faster triaging.

When opening the issue, please include the following information (where relevant):

### System Information
- OS:
- How you installed the benchmark (docker, conda, source):
- Python version (`python --version` output):
- CUDA/cuDNN version:
- NVIDIA driver version:
- Conda version (if using conda):
- Docker image (if using docker):
- GCC/GXX version (if compiling from source):
- Commit hash of our rep (if compiling from source):

In addition, including the following information will also be very helpful for us to diagnose the problem:
- A script to reproduce the issue (highly recommended if its a build issue)
- Error messages and/or stack traces of the issue (create a gist)
- Context around what you are trying to do

### Results obtained vs Results expected
What results you got vs what results were expected?
