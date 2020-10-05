* Installation

0. Verify that you have installed [Anaconda](https://www.anaconda.com/products/individual) (NOT Miniconda) and you are familiar with the way it works. 
1. Go to [the PyTorch installation page](https://pytorch.org/get-started/locally/) and use the interactive pane to get the correct builds for your CUDA compute capability. You can use [this table](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) to match your GPU and to its CUDA compute capability. This will allow you to utilize your GPU to create BERT embeddings rather than just your CPU.
2. `cd` to the directory that contains `sfake.py`.
3. Install `torch` and `torchvision` using the command obtained above.
4. Use `conda install --file requirements_conda.txt` to install the dependencies that are available on Anaconda. We do this because Conda manages these packages along with their dependencies *with respect to the other packages in the environment*.
5. Finally, use `pip install -r requirements_pip.txt` to install the dependencies only available on PyPI.
6. At this point you should be able to invoke the CLI. Try `python sfake.py -V` or just `python sfake.py` to verify that it's working.

* Troubleshooting

If you encounter an error with `numpy` stating that it could not find a certain `.dll`, try adding `<your Anaconda directory>\Library\bin` to your `PATH` environment variable. This error hints that either your `numpy` installation was corrupted or it simply cannot find where it installed its own libraries for whatever reason. Try to place it near the top of the `PATH` if you can. If that does not work, try `conda uninstall numpy` and then reinstall it with `conda install numpy`.

Be aware that if you choose to install this tool in a `conda` environment other than `base`, you will need to `conda activate` that environment every time you want to use the tool.
